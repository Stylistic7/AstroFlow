# Plan: atomic product publication

**Status:** proposed, 2026-08-14
**Origin:** γ Cyg mosaic served 36,150 tile errors because its `Linear RGB`
master existed on disk without a tile pyramid.

## The incident, precisely

```
07:45:53   combine_broadband_rgb writes the master        20.2 GB
07:45–07:59  pyramid builds (a ~15 min second pass)
07:59:58   `make stop` → pkill -9 kills it at 73%         14.8 GB .tmp orphaned
```

For those 14 minutes the product was **resolvable but untileable**: the
master had replaced the previous (working) one, while its own pyramid did
not yet exist. Every tile request hit `w <= 0` inside the `try` in
`_serve_tile_response`, where `abort(404)` raises `NotFound`, gets caught by
`except Exception`, logged, and returned as **500**. A 500 reads as "retry",
so the viewer looped at ~3,220 req/min until the tab was closed — burying all
pipeline telemetry in the log.

Three independent defects:

1. **A product becomes visible before it is complete.** The payload is
   published by the op; the pyramid is built afterwards by the decorator.
2. **A missing tile is reported as a server error**, which invites retries.
3. **Non-atomic publication in several producers** can leave a truncated file
   at a final content-addressed path.

## The invariant we want

> A file at its final content-addressed path is complete, and so is every
> artifact derived from it.

Today that is false in two ways: the pyramid may be absent (defect 1), and
`shutil.copy` publishers can leave torn files (defect 3).

## Sequencing constraints (these are load-bearing)

* **Atomicity must land before the stamp is reduced.** The `CONTHASH` header
  stamp currently doubles as a torn-write detector: a half-copied file has no
  readable header, `get_content_hash` falls back to `hash_of_image_data`, the
  hash mismatches, and the node recomputes. Remove the stamp before making
  publication atomic and that safety net disappears.
* **Deferred publish (3) must land before stamp-at-first-write (4).** Today a
  stamp implies the decorator ran past the body. If stamping moves into the
  op's first write while publication is still immediate, a killed build
  leaves a *stamped* payload with no pyramid — a product that looks complete
  and is not.
* **Do NOT reduce `preflight._is_satisfied` to `expected.exists()`.** The
  invariant only holds for files written under the new regime; every existing
  product predates it. Tempting, out of scope, would trust old torn files.

## Phase 1 — web-server defensive fixes

Independent of everything else; fixes the user-visible symptom today.

* **Don't serve tiles by cropping the payload itself.** This changes ONE case
  — a pyramid-needing deliverable whose pyramid is missing. Deliverables that
  have a pyramid (the normal case: 245/245 `post_process_stack` and 295/295
  `stack_files` masters, and every healthy tap) are untouched and keep serving
  tiles from it.

  Today `resolve_product` → `viewable_from_master` falls back to the master
  when no pyramid exists. That is how a 19 GB *stripped* TIFF came to be
  tiled: `_pyramid_levels` returned 1, so `page = max_zoom - z` went negative,
  and pyvips — instead of raising — logged `GLib-GObject-CRITICAL: value "-8"
  ... out of range for property 'page'` and silently served **full resolution
  at every zoom level**. The same fallback caused the 822-second
  `channelStats` requests.

  `needs_display_pyramid()` already encodes the right distinction: float /
  linear deliverables need a pyramid; already-display-ready uint8 `.tiff`s
  (Subframe previews) legitimately serve directly. So `viewable_from_master`
  returns `None` — surfacing as "not yet built" — only when
  `needs_display_pyramid(master)` is true AND the pyramid is absent.

  (Intermediates such as `stack_sho_channels`, which carry no `display=`,
  have no pyramid by design and are not taps — unaffected.)
* `web_server/backend/tiles.py:_serve_tile_response` — re-raise
  `werkzeug.exceptions.HTTPException` before the generic `except Exception`,
  so an out-of-bounds tile returns an honest **404** instead of 500. Only
  genuine failures become 500.
* Rate-limit the error print (count + periodic summary, keyed by
  object/display). One stuck viewer must never be able to bury `### node_exec`
  telemetry again — that is what made the pipeline look idle during triage.
* Frontend: treat 404 as "absent, stop asking" and re-fetch `tile_metadata`
  rather than looping on stale dimensions.

**Test:** a request beyond image bounds returns 404, not 500; N such requests
emit ≤ 1 log line per interval.

## Phase 2 — uniform atomic publication

`rc_astro_cli.py:97-110` already does this correctly, with a comment
explaining the cross-filesystem subtlety:

```python
shutil.copy2(produced, staging)   # stage ONTO the destination's filesystem
os.replace(staging, dest)         # atomic same-fs publish
```

Apply that pattern to every producer that currently publishes with a bare
copy:

| site | current |
|---|---|
| `pix/recipe.py:444` (`copy_output`) | `shutil.copy2(src, dest)` |
| `pix/recipe.py:314` | `shutil.copy2(src, dest)` |
| `mosaics/merging/pixinsight.py:86` | `shutil.copy(...)` |
| `targets/merging/oss.py:61` | `shutil.copy(...)` |
| `targets/merging/pixinsight.py:64` | `shutil.copy(...)` |
| `file_io/generic.py:149-150` | copies file **and** sidecar |

Notes:

* Staging must be **in the destination directory** (not `/tmp`) or
  `os.replace` fails across filesystems — the reason rc-astro does it this way.
* `file_io/generic.py:149-150` publishes two files (payload + sidecar); order
  sidecar first, payload last, same rule as pyramid-before-payload.
* Extract one helper (`publish_atomically(src, dest)`) rather than repeating
  the idiom six times.

**Risk — disk space.** Staging transiently doubles peak usage for the file
being published. On gigapixel XISF that is tens of GB on a volume already
described as dangerously full. **Run Phase 5 first** to reclaim headroom.

**Risk — completeness.** The invariant is only worth anything if it is
universal; one missed publisher and later phases rest on a false premise. Add
a test that greps the tree for `shutil.copy*` whose destination is an op
output path, and fails on new occurrences.

**Audit:** confirm PixInsight never writes directly to a final path.
`_output_path_in_tmp` suggests all PI saves land in `tmp_dir` first, but this
must be verified, not assumed.

## Phase 3 — pyramid lands before payload (fully declarative)

**No implicit state.** No contextvars, no changes to `file_io`, no changes to
any op. The decorator already knows everything it needs *before the body runs*:

* `expected = op.filename_fn(*args, **kwargs)` — the output path. This is
  already relied on by `process_in_panels` ("the crop's stem is its call hash,
  which is exactly the content_hash its result will carry").
* `makes_tiff` — whether this op produces a display deliverable.
* the stamps — all derived from the arguments.

So `products/registry.py`'s wrapper owns the whole publication sequence:

```python
result = func(*args, **kwargs)                    # op writes payload to `expected`
staging = expected.with_name(f".{expected.name}.staging")
os.replace(expected, staging)                     # unpublish (atomic, instant)
try:
    default_make_tile_pyramid(staging, pyr_tmp)
    os.replace(pyr_tmp, pyramid_path_for(expected))   # PYRAMID LANDS FIRST
except Exception:
    log.warning("tile_pyramid_build_failed", ...)     # non-fatal, see below
finally:
    os.replace(staging, expected)                     # PAYLOAD LANDS LAST
```

**Invariant established:** payload present ⇒ pyramid present. That is what
makes existence a genuine completion signal, and what Phase 4 rests on.

**Residual window (accepted).** Between the op's own `os.replace` inside
`_atomic_tiffsave` and the decorator's unpublish, the payload is briefly
visible without a pyramid — microseconds, versus ~15 minutes today. Closing
that last sliver would require the write path to know the publication was
deferred (i.e. implicit state); explicitly rejected as not worth it.

**Continuity during a rebuild is preserved anyway.** The OLD `_tiles.tiff`
stays in place for the whole build, and tile serving resolves through the
pyramid — so viewers keep seeing the previous image until the new pyramid
lands atomically. Combined with Phase 1, a first-ever build shows
"not yet built" rather than a broken image.

**The property this buys:** on a *rebuild*, the previous payload and pyramid
keep serving for the entire build, and are replaced only when the new pair is
complete. The γ Cyg page would never have gone black, and a `pkill` mid-build
would have cost only wasted CPU.

**Risks:**

* **Pyramid failure must stay non-fatal.** `registry.py:375-387` deliberately
  swallows it — *"failing here would throw away the (expensive) master the
  body just produced."* The pyramid build needs its own inner `try` that logs
  `tile_pyramid_build_failed`; the payload republish MUST happen in a
  `finally`, or an OOM on a 20 GB image would strand the master at its
  `.staging` name and the product would vanish entirely.
* **The unpublish/republish pair must be crash-symmetric.** If the process
  dies between the two renames, the payload sits at `.staging` and the product
  is absent. That is recoverable (no cache entry ⇒ the op re-runs) and is the
  correct direction to fail, but the janitor must recognise `.staging` as
  sweepable and startup must not mistake it for a live product.
* **`expected` must match what the op actually wrote.** The decorator derives
  the path from `filename_fn`; if an op writes somewhere else, the unpublish
  finds nothing. Fail loudly (assert the path exists after the body) rather
  than silently skipping the pyramid.
* **More orphaned temps on kill.** Acceptable (a killed build already leaves
  no cache entry and re-runs) but the janitor must sweep both `*.tmp` and
  hidden `.*.tmp`.
* **Long-running writes.** For a 20 GB deliverable the publish step now spans
  the pyramid build; any caller holding a lock or timeout must tolerate it.

**Tests:** killed-before-commit leaves nothing at the final path and no cache
entry; a pyramid build that raises still publishes the payload; during a
rebuild the old product remains readable until the new one commits.

## Phase 4 — stamp at first write

`call_hash`, `STBLHASH` and `image_type` are all derived from the *arguments*,
so they are knowable before the body runs; they are merely computed afterwards
in `registry.py:266`. `write_tiff_image` already serializes a `metadata` dict
into the TIFF description before saving, so the stamps can ride along in the
op's single write.

This deletes `_stamp_header`'s re-encode for files we write ourselves.
`update_tiff_header`'s own docstring concedes it *"re-encodes the file — there
is no in-place tag edit"*.

**Measured 2026-08-14:** `update_tiff_header` on a 0.41 GB float TIFF took
**25.2 s** (0.02 GB/s). It is not an I/O copy — it is a full AdobeDeflate
decompress + recompress of every pixel, so it is CPU-bound and a faster disk
does not help. Extrapolated to the 19.3 GB γ Cyg `Linear RGB` master that is
**~20 minutes to add one header key**. (Estimate: the probe ran on the boot
SSD, not the Astro volume, and 47x extrapolation is not guaranteed linear.)

This also reinterprets the incident timeline. The master's 07:45:53 mtime is
almost certainly when the *stamp's* `os.replace` landed — `update_tiff_header`
rewrites the file, and the rename resets mtime — not when the op finished. So
the post-body work for that product was roughly **35 minutes: ~20 stamping,
~15 building the pyramid**. Stamping at first write nearly halves the window
during which a product is vulnerable, on top of removing the write itself.

* Keep the post-hoc stamp as a fallback for foreign producers (PixInsight,
  rc-astro) that do not route through `write_tiff_image`. The decorator can
  tell them apart declaratively: after the body, read the header of `expected`
  and stamp only if the keys are absent. Our own writes arrive pre-stamped and
  cost nothing; foreign outputs take the existing re-encode path.
* Preserve fill-if-absent semantics: today `IMAGETYP`/`STBLHASH` are only set
  when absent, while `CONTHASH` is always written. Merging into `metadata` at
  write time inverts precedence unless stated explicitly.
* TIFF only initially. FITS/XISF have the same shape but different re-encode
  costs; measure before extending.

**Test:** a file written by an op carries `CONTHASH` == its stem after exactly
one write (assert no second write via mtime/inode); a PixInsight-produced file
still gets stamped.

## Phase 5 — sweep stale temps (do first)

56.6 GB of `.tmp` files currently sit on the Processed volumes; measured by
growth, **43.9 GB across 3 files is stale** and 2 are actively being written.

* Sweep must match **both** `*.tmp` and hidden `.*.tmp` — the earlier count
  missed the dotfile form and undercounted 5 files as 1.
* **Never delete a growing temp.** Gate on "untouched for > N minutes",
  confirmed by a second stat, not a single mtime read. (Killing live work is
  precisely how this incident became permanent.)
* Fold into the existing wall-clock janitor rather than adding a new sweeper.

## What this does not fix

* The pyramid still takes ~15 minutes for a 20 GB master. Phases 3–4 remove
  the *window* and one full re-encode; they do not make the build faster.
  Tiling the master, or reusing its levels, is a separate investigation.
* Products already on disk were published under the old regime. The invariant
  applies to new writes only.

## Suggested order

`5 → 1 → 2 → 3 → 4`

5 first for disk headroom; 1 is independent and fixes the visible symptom; 2
establishes the invariant; 3 depends on 2; 4 depends on 3.

Each phase is independently shippable and independently revertible. Full test
suite must be green before each deploy, which requires the app stopped
(pytest resets the shared Ray cluster).
