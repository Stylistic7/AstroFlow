import { useEffect, useRef, useState, useLayoutEffect, useContext } from 'react';
import * as PIXI from 'pixi.js';
import { Stage, useApp } from '@pixi/react';
import { useParams, useNavigate } from 'react-router-dom';
import { Viewport } from 'pixi-viewport';

import AstroFlowContext from './AstroFlowContext.jsx';
import { buildUrl, buildApiUrl, inheritView } from './urls.js';
import { fetchLinearTile, uploadFloat32Tile } from './linear_tile_loader.js';
import { createToneMappingShader } from './tone_mapping_shader.js';
import { computeAutoStretchParams } from './auto_stretch.js';
import ToneMappingControls, { useHdrCapability } from './ToneMappingControls.jsx';


const TILE_SIZE = 256;


/**
 * Build a unit-quad Mesh with our tone-mapping shader. Each visible
 * tile gets its own mesh — they share the shader instance (so slider
 * updates propagate to all tiles via shared uniforms) but each
 * carries its own uTileTexture and per-tile world-position uniforms.
 *
 * The Mesh replaces PIXI.Sprite from the previous design: a Sprite's
 * built-in shader assumes pre-stretched RGBA tiles, which is exactly
 * what we no longer have. The custom shader needs Mesh-with-geometry
 * to be invoked at all.
 */
function createTileMesh(shader, tileWidth, tileHeight) {
    const verts = new Float32Array([
        0, 0,                       // top-left
        tileWidth, 0,               // top-right
        tileWidth, tileHeight,      // bottom-right
        0, tileHeight,              // bottom-left
    ]);
    const uvs = new Float32Array([
        0, 0,
        1, 0,
        1, 1,
        0, 1,
    ]);
    const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
    const geometry = new PIXI.Geometry()
        .addAttribute('aVertexPosition', verts, 2)
        .addAttribute('aTextureCoord', uvs, 2)
        .addIndex(indices);
    return new PIXI.Mesh(geometry, shader);
}

// Match the HMS / DMS formatting in suggestions/export.py so polygon
// labels read identically to the Telescopius / text export.
const formatRaHms = (raDeg) => {
  const raHours = ((raDeg % 360) + 360) % 360 / 15;
  const h = Math.floor(raHours);
  const mFloat = (raHours - h) * 60;
  const m = Math.floor(mFloat);
  const s = (mFloat - m) * 60;
  return `${String(h).padStart(2, '0')}h${String(m).padStart(2, '0')}m${s.toFixed(1).padStart(4, '0')}s`;
};

const formatDecDms = (decDeg) => {
  const sign = decDeg < 0 ? '-' : '+';
  const absDec = Math.abs(decDeg);
  const d = Math.floor(absDec);
  const mFloat = (absDec - d) * 60;
  const m = Math.floor(mFloat);
  const s = (mFloat - m) * 60;
  return `${sign}${String(d).padStart(2, '0')}°${String(m).padStart(2, '0')}'${s.toFixed(0).padStart(2, '0')}"`;
};

const TileViewer = ({ objectType: objectTypeProp, imageHash: imageHashProp, filter: filterProp, pipeline, display, showPolygons, showPanelNumbers }) => {
  const params = useParams();
  const { objectType: objectTypeParam, imageHash: imageHashParam, filter: filterParam } = params;

  const objectType = objectTypeProp ?? objectTypeParam;
  const imageHash = imageHashProp ?? imageHashParam;
  const filter = filterProp ?? filterParam;

  const [metadata, setMetadata] = useState(null);
  const [tileMetadata, setTileMetadata] = useState(null);
  const containerRef = useRef(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });

  // The auto-stretch is not parameterized — `previews.operations.autostretch`
  // hard-codes the STF parameters and the shader applies them per
  // pixel. The only user-facing knob in this panel is the HDR
  // brightness multiplier, gated on display capability.
  const [peakBrightness, setPeakBrightness] = useState(2.0);
  const hdrCapable = useHdrCapability();
  const [hdrEnabled, setHdrEnabled] = useState(false);

  const { mosaics, stacks, targets, showQuarantined, isQuarantined, computeGraphStatus, observingPlan } = useContext(AstroFlowContext);

  const navigate = useNavigate();

  useLayoutEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(() => {
      const bounds = containerRef.current.getBoundingClientRect();
      setContainerSize({ width: bounds.width, height: bounds.height });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // The view triple is all-or-nothing — see [[project_url_grammar]].
  // Without it we can't request a tap-keyed artifact.
  const view = (pipeline && display && filter)
    ? { pipeline, display, filter }
    : null;

  useEffect(() => {
    if (!view) return;
    const url = buildApiUrl(
      '/api/metadata',
      [{ type: objectType, id: imageHash }],
      view,
    );
    fetch(url)
      .then((res) => res.ok ? res.json() : null)
      .then((meta) => { if (meta) setMetadata(meta); })
      .catch((e) => console.warn(e));
  }, [objectType, imageHash, view?.pipeline, view?.display, view?.filter, observingPlan]);

  useEffect(() => {
    if (!view) return;
    const url = buildApiUrl(
      '/tap/tile_metadata',
      [{ type: objectType, id: imageHash }],
      view,
    );
    fetch(url)
      .then((res) => res.ok ? res.json() : null)
      .then((meta) => { if (meta) setTileMetadata(meta); });
  }, [objectType, imageHash, view?.pipeline, view?.display, view?.filter, computeGraphStatus]);

  const onClick = (poly, event) => {
    const e = event.nativeEvent;

    // Hierarchy: the parent mosaic (if any) → drilldown target type +
    // its id. Stacks and the target-with-no-matching-stack case nest
    // identically; the only oddity is target → stack auto-resolution
    // by the active filter.
    const hierarchy = params.mosaic_id
      ? [{ type: 'mosaic', id: params.mosaic_id }]
      : [];
    if (poly.type === 'stack' && poly.target_id) {
      hierarchy.push({ type: 'target', id: poly.target_id });
      hierarchy.push({ type: 'stack', id: poly.stable_hash });
    } else if (poly.type === 'target' && filter && targets?.[poly.stable_hash]) {
      hierarchy.push({ type: 'target', id: poly.stable_hash });
      const target = targets[poly.stable_hash];
      const stackId = target.stack_ids?.find(s => stacks?.[s]?.filter === filter);
      if (stackId) hierarchy.push({ type: 'stack', id: stackId });
    } else {
      hierarchy.push({ type: poly.type, id: poly.stable_hash });
    }

    const destType = hierarchy[hierarchy.length - 1].type;
    const url = buildUrl({
      hierarchy,
      view: view ? inheritView(view, destType) : null,
    });

    if (e.metaKey || e.ctrlKey) {
      window.open(url, '_blank');
    } else {
      navigate(url);
    }
  };

  if (!metadata || !tileMetadata || !containerRef.current) return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;

  // The viewport's world is the image itself, not the power-of-two-
  // rounded tile-grid extent. The old `tileWidth * 2^maxZoom` formula
  // (e.g. 256 × 32 = 8192 for a 6026×5841 image) padded the world with
  // dead space past the image edge, and pixi-viewport's first-tick
  // recenter pass would slide the viewport center to `worldWidth/2 =
  // 4096` — visibly off-image — instead of `imageWidth/2 = 3013`.
  const worldWidth = tileMetadata.width;
  const worldHeight = tileMetadata.height;

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Stage
        width={containerSize.width}
        height={containerSize.height}
        options={{
          backgroundColor: 0x000000,
          resizeTo: containerRef.current,
        }}
        style={{ width: '100%', height: '100%' }}
      >
        <ImperativeViewport
          objectType={objectType}
          imageHash={imageHash}
          view={view}
          port={tileMetadata.port}
          imageWidth={tileMetadata.width}
          imageHeight={tileMetadata.height}
          worldWidth={worldWidth}
          worldHeight={worldHeight}
          maxZoom={tileMetadata.maxZoom}
          mtime={tileMetadata.mtime}
          showPolygons={showPolygons}
          showPanelNumbers={showPanelNumbers}
          metadata={metadata}
          mosaics={mosaics}
          onClick={onClick}
          showQuarantined={showQuarantined}
          isQuarantined={isQuarantined}
          stacks={stacks}
          peakBrightness={hdrEnabled ? peakBrightness : 1.0}
          hdrEnabled={hdrEnabled}
        />
      </Stage>
      <ToneMappingControls
        peakBrightness={peakBrightness}
        setPeakBrightness={setPeakBrightness}
        hdrEnabled={hdrEnabled}
        setHdrEnabled={setHdrEnabled}
        hdrCapable={hdrCapable}
      />
    </div>
  );
};


/** Draw an exposure-duration clock face — same dial style as
    ExposureDurationWidget — into a PIXI Graphics, centered at (cx, cy)
    with the given pixel radius.

    `hours` (blue wedge): total raw integration across every subframe,
    upper bound. Clamped to [0, 12).

    `wbppHours` (green wedge, optional): the value WBPP actually used
    after its own rejection pass (XISF PCL:TotalExposureTime). Drawn
    on top of the blue wedge so the green visible region is "what was
    kept" and the surrounding blue is "what was rejected." When null,
    only the blue wedge renders. */
function drawExposureClock(g, cx, cy, radius, hours, wbppHours = null) {
  // Background disc.
  g.beginFill(0x333333);
  g.drawCircle(cx, cy, radius);
  g.endFill();

  // Blue wedge for the raw total integration.
  const fraction = Math.min(Math.max(hours, 0), 11.999) / 12;
  if (fraction > 0) {
    const start = -Math.PI / 2;
    const end = start + 2 * Math.PI * fraction;
    g.beginFill(0x4278f5);
    g.moveTo(cx, cy);
    g.arc(cx, cy, radius, start, end);
    g.lineTo(cx, cy);
    g.endFill();
  }

  // Green wedge for the WBPP-effective integration, drawn ON TOP.
  if (wbppHours != null && wbppHours > 0) {
    const wbppFraction = Math.min(Math.max(wbppHours, 0), 11.999) / 12;
    const start = -Math.PI / 2;
    const end = start + 2 * Math.PI * wbppFraction;
    g.beginFill(0x4caf50);
    g.moveTo(cx, cy);
    g.arc(cx, cy, radius, start, end);
    g.lineTo(cx, cy);
    g.endFill();
  }

  // 12 tick marks.
  g.lineStyle(radius * 0.05, 0xffffff, 1);
  for (let h = 0; h < 12; h++) {
    const theta = (2 * Math.PI * h / 12) - Math.PI / 2;
    g.moveTo(cx + radius * 0.75 * Math.cos(theta), cy + radius * 0.75 * Math.sin(theta));
    g.lineTo(cx + radius * Math.cos(theta), cy + radius * Math.sin(theta));
  }

  // Outer ring.
  g.lineStyle(radius * 0.18, 0xffffff, 1);
  g.drawCircle(cx, cy, radius);
  g.lineStyle(0);
}


const ImperativeViewport = ({ objectType, imageHash, view, port, imageWidth, imageHeight, worldWidth, worldHeight, maxZoom, mtime, showPolygons, showPanelNumbers, metadata, mosaics, onClick, showQuarantined, isQuarantined, stacks, peakBrightness, hdrEnabled }) => {
  const app = useApp();

  // HDR canvas reconfiguration. Only fires when HDR is being toggled
  // ON — calling `canvas.configureHighDynamicRange({mode: 'default'})`
  // on every mount (the previous behavior) re-initializes the canvas
  // context on some browsers, which nulls out PIXI's EventSystem
  // `domElement` and breaks the first wheel event with
  // `Cannot read properties of null (reading 'isConnected')`. The
  // SDR default doesn't need re-asserting on a canvas that never had
  // HDR engaged in the first place.
  useEffect(() => {
    if (!hdrEnabled) return;
    const canvas = app?.renderer?.view;
    if (!canvas) return;
    if (typeof canvas.configureHighDynamicRange === 'function') {
      try {
        canvas.configureHighDynamicRange({ mode: 'extended' });
      } catch (e) {
        console.warn('HDR canvas reconfigure failed:', e);
      }
    }
    app.renderer.render(app.stage);
    return () => {
      if (typeof canvas.configureHighDynamicRange === 'function') {
        try { canvas.configureHighDynamicRange({ mode: 'default' }); } catch {}
      }
    };
  }, [hdrEnabled, app]);
  const viewportRef = useRef(null);
  const spritesRef = useRef({});
  // The single tone-mapping shader instance shared by every tile mesh
  // for this image. Created once per image-open; updated uniforms
  // (LUT, slider values) propagate to all meshes automatically.
  const shaderRef = useRef(null);
  // The 1D LUT texture the shader binds as uLutTexture. Default is
  // an identity LUT (no stretch); the auto-stretch optimizer (Phase
  // C5) replaces its contents from ImageStatistics.
  // Survives across view-triple changes (display/pipeline/filter pill
  // switches): when the main viewport-setup effect cleans up + re-runs,
  // we save center+scale here so the new viewport restores the user's
  // zoom/pan instead of resetting to fit-to-screen. Cleared explicitly
  // when imageHash changes (different object → semantically different
  // viewport, fit-to-screen is the right default).
  const savedViewStateRef = useRef(null);
  const lastImageHashRef = useRef(imageHash);
  if (lastImageHashRef.current !== imageHash) {
    savedViewStateRef.current = null;
    lastImageHashRef.current = imageHash;
  }

  // Bookkeeping for the polygon-overlay effect below. Tracks every
  // pixi child added by polygon rendering so we can clear them on
  // re-render without rebuilding the viewport (which would tear down
  // the wheel listener and panel state we just spent effort fixing).
  // `viewportVersion` ticks every time a fresh viewport is created;
  // the polygon effect listens for it so polygons get re-added when
  // the viewport reincarnates (image change, etc.).
  const polygonRefsRef = useRef([]);
  const [viewportVersion, setViewportVersion] = useState(0);

  // When the tile pyramid file changes (new merge completed), clear
  // the sprite cache so tiles are re-fetched with the new content.
  // Also destroys the tile textures we uploaded to the GPU, so the
  // memory frees properly between merge cycles.
  useEffect(() => {
    if (!viewportRef.current) return;
    Object.values(spritesRef.current).forEach(s => {
      const tex = s._tileTexture;
      s.parent?.removeChild(s);
      s.destroy();
      // Destroy the BaseTexture too; PIXI.Mesh.destroy doesn't
      // tear down per-instance textures by default.
      if (tex) tex.destroy(true);
    });
    spritesRef.current = {};
  }, [mtime]);

  // One shader instance per image-open, shared across every tile mesh.
  // The auto-stretch parameters (E1, E5, max) are scalar uniforms — no
  // LUT, no context map, no slider state. The previous LUT-based
  // implementation introduced a quantization artifact at the E1
  // boundary: with only 256 LUT entries spread over [0, 1] and E1 ≈
  // sky_median ≈ 0.146, every entry below E1 was 0; sky pedestal pixels
  // sampled LINEAR-filtered between two zero entries and crushed to
  // black. The shader now evaluates the MTF closed form per pixel.
  useEffect(() => {
    if (!view) return;
    shaderRef.current = createToneMappingShader();
    return () => {
      shaderRef.current = null;
    };
  }, [objectType, imageHash, view?.pipeline, view?.display, view?.filter, mtime, port]);

  // Push the autostretch parameters into the shader whenever the
  // ImageStatistics arrive. The values come from `computeAutoStretchParams`,
  // which mirrors `previews.operations.autostretch` exactly.
  useEffect(() => {
    if (!metadata?.statistics || !shaderRef.current) return;
    const { E1, E5, identity } = computeAutoStretchParams(metadata.statistics);
    shaderRef.current.uniforms.uMax = metadata.statistics.max_value || 1.0;
    shaderRef.current.uniforms.uE1 = identity ? 0.0 : E1;
    shaderRef.current.uniforms.uE5 = identity ? 0.5 : E5;
    app.renderer.render(app.stage);
  }, [metadata?.statistics, app]);

  // HDR brightness multiplier; the only per-frame scalar that changes
  // post-image-open. Slider state lives at the TileViewer root.
  useEffect(() => {
    if (!shaderRef.current) return;
    shaderRef.current.uniforms.uPeakBrightness = peakBrightness;
    app.renderer.render(app.stage);
  }, [peakBrightness, app]);

  useEffect(() => {
    const screenWidth = app.renderer.view.clientWidth;
    const screenHeight = app.renderer.view.clientHeight;

    const viewport = new Viewport({
      screenWidth,
      screenHeight,
      worldWidth,
      worldHeight,
      events: app.renderer.events,
      passiveWheel: false,
    });

    viewport.drag().pinch().decelerate().wheel();

    // Capture the canvas this viewport's wheel listener was attached
    // to. By the time cleanup runs, `app.renderer.events.domElement`
    // may have been set to null by `EventSystem.destroy()` (StrictMode
    // unmount, app re-create on Stage prop change, …). Without a
    // captured reference, `viewport.destroy()` silently fails to call
    // `removeEventListener`, the old wheel listener stays bound on the
    // (reused) canvas element, and every subsequent scroll fires both
    // the live viewport's handler AND every orphaned dead one.
    const wheelCanvas = app.renderer.events?.domElement;
    const wheelFunction = viewport.input?.wheelFunction;

    // Wrap pixi-viewport's wheel handler so any stale invocation
    // through an orphaned listener (during the brief window before
    // cleanup catches up) returns silently instead of crashing inside
    // `events.mapPositionToPoint` on a null `domElement`. With the
    // listener-cleanup fix below this should rarely fire, but it's a
    // cheap safety net.
    if (viewport.input && typeof viewport.input.handleWheel === "function") {
      const origHandleWheel = viewport.input.handleWheel.bind(viewport.input);
      viewport.input.handleWheel = function (e) {
        if (!viewport.options.events?.domElement) return;
        return origHandleWheel(e);
      };
    }

    const fitScale = Math.min(
      screenWidth / imageWidth,
      screenHeight / imageHeight
    );

    viewport.clampZoom({ minScale: fitScale, maxScale: 1 });
    const saved = savedViewStateRef.current;
    if (saved) {
      viewport.setZoom(saved.scale, true);
      viewport.moveCenter(saved.centerX, saved.centerY);
    } else {
      viewport.setZoom(fitScale, true);
      viewport.moveCenter(imageWidth / 2, imageHeight / 2);
    }

    viewportRef.current = viewport;
    app.stage.addChild(viewport);

    const updateClamp = () => {
      const scale = viewport.scale.x;
      const halfWidth = screenWidth / (2 * scale);
      const halfHeight = screenHeight / (2 * scale);

      viewport.clamp({
        left: -halfWidth,
        right: imageWidth + halfWidth,
        top: -halfHeight,
        bottom: imageHeight + halfHeight,
        // When the image is smaller than the visible region (zoomed
        // out far enough that the whole mosaic fits), pixi-viewport's
        // default underflow is 'topleft' — which is why the mosaic
        // sticks to the left and you can't pan it to the middle. We
        // want it visually centered instead.
        underflow: 'center',
      });
    };


    const updateTiles = () => {
      const scale = viewport.scale.x;
      const pixelsPerScreenPixel = 1 / scale;
      let bestZ = Math.ceil(maxZoom - Math.log2(pixelsPerScreenPixel));
      bestZ = Math.max(0, Math.min(maxZoom, bestZ));

      const levelScale = Math.pow(2, maxZoom - bestZ);
      const scaledWidth = Math.ceil(imageWidth / levelScale);
      const scaledHeight = Math.ceil(imageHeight / levelScale);

      const tilesX = Math.ceil(scaledWidth / TILE_SIZE);
      const tilesY = Math.ceil(scaledHeight / TILE_SIZE);

      const bounds = viewport.getVisibleBounds();
      const xStart = Math.floor(bounds.x / (TILE_SIZE * levelScale));
      const yStart = Math.floor(bounds.y / (TILE_SIZE * levelScale));
      const xEnd = Math.ceil((bounds.x + bounds.width) / (TILE_SIZE * levelScale));
      const yEnd = Math.ceil((bounds.y + bounds.height) / (TILE_SIZE * levelScale));

      for (let x = xStart; x < xEnd; x++) {
        for (let y = yStart; y < yEnd; y++) {
          if (x < 0 || y < 0 || x >= tilesX || y >= tilesY) continue;
          const key = `${bestZ}/${x}/${y}`;
          if (spritesRef.current[key]) continue;

          if (!view || !shaderRef.current) continue;
          // New URL suffix: raw float32 bytes, brotli-encoded by the
          // server. `.png` doesn't apply — the data is no longer a
          // browser-renderable image format. The fetch goes through
          // `fetchLinearTile` which returns the decoded Float32Array.
          const relativePath = buildApiUrl(
            '/tap/tile',
            [{ type: objectType, id: imageHash }],
            view,
            `/${bestZ}/${x}/${y}.f32`,
          );
          const url = new URL(relativePath, `${window.location.protocol}//${window.location.hostname}:${port}`);
          if (mtime) url.searchParams.set('v', mtime);

          fetchLinearTile(url.href)
            .then((result) => {
              if (!result || !shaderRef.current) return;
              // Server pads every tile to TILE_SIZE × TILE_SIZE with
              // zeros for edge tiles, so the texture is always the
              // same size as the mesh and the shader's zero short-
              // circuit handles "past the image bound" correctly.
              const tileTex = uploadFloat32Tile(
                result.array, result.width, result.height,
              );
              const mesh = createTileMesh(
                shaderRef.current,
                TILE_SIZE * levelScale, TILE_SIZE * levelScale,
              );
              mesh.x = x * TILE_SIZE * levelScale;
              mesh.y = y * TILE_SIZE * levelScale;
              mesh.zIndex = bestZ;
              // Per-mesh shader uniforms are not really a thing in
              // PIXI's stock Shader model — uniforms live on the
              // Shader instance and are shared. To bind the right
              // tile texture per draw, we override at draw time by
              // attaching the texture as a mesh property and reading
              // it via shader.uniforms before the mesh renders. The
              // render order is deterministic (PIXI iterates children
              // in zIndex order), so this works without a custom
              // PIXI.MeshMaterial subclass.
              mesh._tileTexture = tileTex;
              // Replace the default render path with one that sets
              // per-tile uniforms just before drawing.
              const origRender = mesh.render.bind(mesh);
              mesh.render = (renderer) => {
                const uniforms = shaderRef.current?.uniforms;
                if (uniforms) {
                  uniforms.uTileTexture = mesh._tileTexture;
                }
                origRender(renderer);
              };

              viewport.addChild(mesh);
              viewport.sortChildren();
              spritesRef.current[key] = mesh;
            })
            .catch((err) => {
              console.warn(`Failed to load tile: ${url}`, err);
            });
        }
      }

      updateClamp();
    };

    // Tiles re-fetch on every move/zoom; polygons don't — they're added
    // as viewport children that pan/zoom with the world automatically,
    // and they only need re-rendering when their source data changes,
    // which is what the separate polygon effect below handles.
    const updateAll = () => { updateTiles(); };

    updateAll();
    viewport.on('moved', updateAll);
    viewport.on('zoomed', updateAll);
    viewport.on('moved-end', updateAll);
    viewport.on('zoomed-end', updateAll);

    // Signal to the polygon effect that a fresh viewport is in place.
    setViewportVersion(v => v + 1);

    return () => {
      // Capture zoom/pan before destroying so the next viewport
      // (re-created when the user changes pipeline/display/filter)
      // restores them. lastImageHashRef clears this when the user
      // navigates to a different object.
      savedViewStateRef.current = {
        scale: viewport.scale.x,
        centerX: viewport.center.x,
        centerY: viewport.center.y,
      };
      viewport.off('moved', updateAll);
      viewport.off('zoomed', updateAll);
      viewport.off('moved-end', updateAll);
      viewport.off('zoomed-end', updateAll);
      // Belt-and-suspenders wheel-listener removal: `viewport.destroy()`
      // tries to do this internally via `events.domElement.removeEventListener`,
      // but if the EventSystem was already destroyed `domElement` is
      // null and the call silently no-ops, leaving the listener bound.
      // Use the captured-at-setup canvas+wheelFunction to remove it
      // directly.
      if (wheelCanvas && wheelFunction) {
        try { wheelCanvas.removeEventListener('wheel', wheelFunction); } catch {}
      }
      if (viewport.parent) {
        viewport.parent.removeChild(viewport);
      }
      try { viewport.destroy({ children: true }); } catch {}
      spritesRef.current = {};
    };
    // Deps deliberately narrowed to things that genuinely require the
    // viewport to be rebuilt (a different image, a different app, a
    // different world size). Polygon-related state — metadata, stacks,
    // mosaics, showPolygons, showPanelNumbers — used to be here, but
    // any of those flipping under the user mid-scroll tore down the
    // viewport (and triggered the leaked-listener pile-up we just
    // fixed). Polygons re-render via `updateAll()` from the viewport's
    // own move/zoom events anyway.
  }, [app, imageHash, view?.pipeline, view?.display, view?.filter, imageWidth, imageHeight, maxZoom, worldWidth, worldHeight]);

  // Polygon / overlay graphics live in the viewport but their content
  // depends on metadata, stacks, and the user-toggleable showPolygons /
  // showPanelNumbers booleans. Re-rendering them via a full viewport
  // rebuild (the original design) tore down the wheel listener and
  // caused the panel to jump back to fit-to-screen on every toggle, so
  // they live in their own effect — clear the previously-added
  // children, then add new ones if showPolygons. Tracked in
  // `polygonRefsRef` so cleanup doesn't need to walk the whole viewport.
  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    for (const obj of polygonRefsRef.current) {
      if (obj.parent === viewport) viewport.removeChild(obj);
      try { obj.destroy({ children: true }); } catch {}
    }
    polygonRefsRef.current = [];
    if (!showPolygons || !metadata?.objects) {
      viewport.sortChildren();
      return;
    }

    const refs = polygonRefsRef.current;
    const add = (obj) => { viewport.addChild(obj); refs.push(obj); };

    const polygons = metadata.objects.filter(poly =>
      showQuarantined || !isQuarantined(poly.type, poly.stable_hash)
    );

    // Per-stack WBPP-effective exposure for the green wedge on the
    // exposure clocks.
    const wbppExposureByStack = {};
    for (const poly of metadata.objects) {
      if (poly.polygon_type === 'output' && poly.wbpp_exposure_s != null) {
        wbppExposureByStack[poly.stable_hash] = poly.wbpp_exposure_s;
      }
    }

    const typeColors = { stack: '#0000ff', target: '#0000ff', mosaic: '#ff0000' };
    const outputColor = '#00cc00';

    for (const poly of polygons) {
      const g = new PIXI.Graphics();
      g.zIndex = 9999;
      const color = poly.polygon_type === 'output'
        ? outputColor
        : (typeColors[poly.type] ?? '#ff0000');
      g.lineStyle(imageWidth / 500, color, poly.polygon_type === 'output' ? 0.6 : 1);
      g.moveTo(poly.points[0][0], poly.points[0][1]);
      for (let i = 1; i < poly.points.length; i++) {
        g.lineTo(poly.points[i][0], poly.points[i][1]);
      }
      g.closePath();
      g.eventMode = 'static';
      g.hitArea = new PIXI.Polygon(poly.points.flat());
      g.cursor = 'pointer';
      g.on('pointertap', (event) => onClick(poly, event));
      const highlight = new PIXI.Graphics();
      highlight.beginFill(0xffffff, 0.15);
      highlight.moveTo(poly.points[0][0], poly.points[0][1]);
      for (let i = 1; i < poly.points.length; i++) {
        highlight.lineTo(poly.points[i][0], poly.points[i][1]);
      }
      highlight.closePath();
      highlight.endFill();
      highlight.visible = false;
      g.addChild(highlight);
      g.on('pointerover', () => { highlight.visible = true; });
      g.on('pointerout', () => { highlight.visible = false; });
      add(g);

      if (poly.polygon_type === 'input') {
        const xs = poly.points.map(p => p[0]);
        const ys = poly.points.map(p => p[1]);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const dotRadius = imageWidth / 150;

        if (poly.has_decisions) {
          const dot = new PIXI.Graphics();
          dot.zIndex = 10000;
          dot.beginFill(0xffa726, 0.9);
          dot.drawCircle(maxX, minY, dotRadius);
          dot.endFill();
          dot.lineStyle(dotRadius * 0.2, 0xffffff, 0.8);
          dot.drawCircle(maxX, minY, dotRadius);
          add(dot);
        }

        const stack = poly.active_stack_hash ? stacks?.[poly.active_stack_hash] : null;
        const totalSeconds = stack
          ? (stack.raw_integration_s ?? stack.integration)
          : null;
        if (typeof totalSeconds === 'number') {
          const clockRadius = imageWidth / 80;
          const offset = poly.has_decisions ? dotRadius * 2.5 : 0;
          const clock = new PIXI.Graphics();
          clock.zIndex = 10000;
          const wbppSeconds = wbppExposureByStack[poly.active_stack_hash];
          drawExposureClock(
            clock,
            maxX - offset - clockRadius,
            minY + clockRadius,
            clockRadius,
            totalSeconds / 3600,
            wbppSeconds != null ? wbppSeconds / 3600 : null,
          );
          add(clock);
        }

        if (showPanelNumbers && typeof poly.panel_number === 'number') {
          const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
          const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
          const fontSize = Math.round(imageWidth / 30);
          const number = new PIXI.Text(String(poly.panel_number), {
            fontFamily: 'sans-serif',
            fontSize,
            fontWeight: 'bold',
            fill: 0xffffff,
            stroke: 0x000000,
            strokeThickness: Math.max(2, fontSize * 0.12),
            align: 'center',
          });
          number.anchor.set(0.5, 0.5);
          number.position.set(cx, cy);
          number.zIndex = 10001;
          add(number);

          if (typeof poly.ra_deg === 'number' && typeof poly.dec_deg === 'number') {
            const coordsSize = Math.round(imageWidth / 90);
            const lineHeight = coordsSize * 1.2;
            const lines = [
              formatRaHms(poly.ra_deg),
              formatDecDms(poly.dec_deg),
            ];
            if (typeof poly.rotation_deg === 'number') {
              lines.push(`PA ${poly.rotation_deg.toFixed(2)}°`);
            }
            if (poly.rise_time_utc) {
              const rise = new Date(poly.rise_time_utc);
              lines.push(`rises ${rise.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`);
            }
            lines.forEach((text, i) => {
              const t = new PIXI.Text(text, {
                fontFamily: 'monospace',
                fontSize: coordsSize,
                fill: 0xffffff,
                stroke: 0x000000,
                strokeThickness: Math.max(1, coordsSize * 0.15),
                align: 'center',
              });
              t.anchor.set(0.5, 0);
              t.position.set(cx, cy + fontSize * 0.6 + i * lineHeight);
              t.zIndex = 10001;
              add(t);
            });
          }
        }
      }
    }

    viewport.sortChildren();
  }, [viewportVersion, metadata, showPolygons, showPanelNumbers, stacks, mosaics, showQuarantined, isQuarantined, imageWidth]);

  return null;
};

export default TileViewer;
