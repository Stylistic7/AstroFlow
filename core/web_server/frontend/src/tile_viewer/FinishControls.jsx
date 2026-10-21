/**
 * One control per served finish param: a switch for `kind === "toggle"`, one
 * colour picker per `Color` knob (its three `kind === "color"` params,
 * suffixed _r/_g/_b), a slider otherwise. Built-in (framework-injected)
 * params come first and are separated from the author's own by a rule.
 * Shared by the tile viewer and the sky viewer: the finish is the same
 * object in both, so its controls are too.
 *
 * `values` are the current param values by name, `onChange(patch)` merges
 * new ones; `seedValues` are what the image-adaptive seed produced (slider
 * ranges centre on these, not the live values, so the track never moves
 * under the user's finger); `channelStats` / `bandRoles` are input 0's
 * (the noise scale of the stretch sliders); `folded` names the collapsed
 * sections and `onToggleSection(title)` flips one.
 */
import Box from '@mui/material/Box';
import ButtonBase from '@mui/material/ButtonBase';
import Collapse from '@mui/material/Collapse';
import FormControlLabel from '@mui/material/FormControlLabel';
import Slider from '@mui/material/Slider';
import Switch from '@mui/material/Switch';
import Typography from '@mui/material/Typography';

import LineColorPicker from './LineColorPicker.jsx';


export default function FinishControls({
    params, values, onChange, seedValues, channelStats, bandRoles, folded, onToggleSection,
}) {
    // `kind: "data"` params are measured from the image (seeded), not controls.
    const shown = params.filter((p) => p.kind !== 'data');
    const builtin = shown.filter((p) => p.builtin);
    const authored = shown.filter((p) => !p.builtin);
    // Stretch values on the pinned gauge are ~0.005: toFixed(3) rendered
    // every black/white point as "0.005" / "0.009" and the readout never
    // moved. Show significant figures instead of fixed decimals.
    const fmtParam = (v) => (Math.abs(v) < 0.1 && v !== 0
        ? Number(v).toPrecision(3) : Number(v).toFixed(3));
    const colorControl = (p) => {
        if (!p.name.endsWith('_r')) return null;   // _g/_b ride on the _r picker
        const base = p.name.slice(0, -2);
        const names = ['_r', '_g', '_b'].map((c) => base + c);
        const value = names.map((n) => values[n]
            ?? params.find((q) => q.name === n)?.default ?? 0);
        return (
            <LineColorPicker
                key={base}
                label={p.label}
                value={value}
                onChange={(rgb) => onChange({ [names[0]]: rgb[0], [names[1]]: rgb[1], [names[2]]: rgb[2] })}
            />
        );
    };
    const control = (p) => (p.kind === 'color' ? colorControl(p) : p.kind === 'toggle' ? (
        <FormControlLabel
            key={p.name}
            control={
                <Switch
                    size="small"
                    checked={(values[p.name] ?? p.default) > 0.5}
                    onChange={(e) => onChange({ [p.name]: e.target.checked ? 1 : 0 })}
                />
            }
            label={<Typography variant="caption" sx={{ opacity: 0.85 }}>{p.label}</Typography>}
            sx={{ ml: 0, color: '#fff' }}
        />
    ) : (
        <Box key={p.name}>
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.85 }}>
                {p.label} — {fmtParam(values[p.name] ?? p.default)}
            </Typography>
            {(() => {
                // STRETCH SLIDERS ARE +/-3 SIGMA WINDOWS around the seeded
                // point (Stylistic7, 2026-09-03). A black point lives within a
                // few sigma of the sky median, so (sky - black) is ~4e-5 on
                // the pinned gauge: any track scaled to the parameter's
                // DECLARED range (0..0.05) stepped tens of sigma and the
                // image died on the smallest drag. Bounding exploration by
                // the image's own noise makes every step a fraction of a
                // sigma, at any depth, and keeps the control meaningful
                // rather than nominal. Non-stretch params keep their
                // declared range (log where they ask for it).
                const SIGMA_SPAN = 3;
                const st = channelStats;
                // A per-band param is suffixed by its band's role; a linked
                // one (no such band) pools every band's sigma.
                const chIdx = bandRoles?.indexOf(p.name.slice(-1)) ?? -1;
                const sigmaOf = () => {
                    if (!st || !st.length) return null;
                    const c = chIdx >= 0 ? st[chIdx] : null;
                    if (c) return c.skySigma ?? c.sigma;
                    return st.reduce((a, x) => a + (x.skySigma ?? x.sigma), 0) / st.length;
                };
                const anchor = seedValues[p.name];
                // BLACK is a NOISE-scale control: it lives within a few
                // sigma of the sky median, so +/-3 sigma is the whole
                // useful world. WHITE is a SIGNAL-scale control: the
                // bright anchor sits ~250 sigma above sky and the useful
                // range runs from just-above-sky to the brightest star, so
                // a +/-3 sigma window moves it 0.9% and does visibly
                // nothing (Stylistic7, 2026-09-04). White therefore gets a
                // GEOMETRIC window around its seed instead.
                const isBlack = /^black/.test(p.name);
                const isWhite = /^white/.test(p.name) || /white$/.test(p.name);
                const sig = isBlack ? sigmaOf() : null;
                if (anchor != null && isWhite) {
                    const lo = Math.max(p.lo, anchor / 4);
                    const hi = Math.min(p.hi, anchor * 6);
                    const toL = (v) => Math.log(Math.min(hi, Math.max(lo, v)));
                    return (
                        <Slider
                            size="small"
                            value={toL(values[p.name] ?? p.default)}
                            min={toL(lo)} max={toL(hi)}
                            step={(toL(hi) - toL(lo)) / 200}
                            onChange={(_, q) => onChange({ [p.name]: Math.exp(q) })}
                            sx={{ color: '#fff' }}
                        />
                    );
                }
                if (anchor != null && sig) {
                    const lo = Math.max(p.lo, anchor - SIGMA_SPAN * sig);
                    const hi = Math.min(p.hi, anchor + SIGMA_SPAN * sig);
                    return (
                        <Slider
                            size="small"
                            value={Math.min(hi, Math.max(lo, values[p.name] ?? p.default))}
                            min={lo} max={hi} step={(hi - lo) / 200}
                            onChange={(_, v) => onChange({ [p.name]: v })}
                            sx={{ color: '#fff' }}
                        />
                    );
                }
                // LOG-SCALED where the param says so: in log space each
                // step is a fixed RATIO, which is what stretch controls
                // want. `lo` is floored off zero (log(0) is undefined) at
                // hi/1000, still far below any real sky.
                const useLog = p.scale === 'log' && p.hi > 0;
                const lo = useLog ? Math.max(p.lo, p.hi / 1000) : p.lo;
                const toPos = (v) => (useLog
                    ? Math.log(Math.min(p.hi, Math.max(lo, v))) : v);
                const fromPos = (q) => (useLog ? Math.exp(q) : q);
                return (
                    <Slider
                        size="small"
                        value={toPos(values[p.name] ?? p.default)}
                        min={toPos(lo)} max={toPos(p.hi)}
                        step={(toPos(p.hi) - toPos(lo)) / 200}
                        onChange={(_, q) => onChange({ [p.name]: fromPos(q) })}
                        sx={{ color: '#fff' }}
                    />
                );
            })()}
        </Box>
    ));
    // Sections: the built-in controls as "Display", then each run of the
    // author's params sharing a group. A titled section folds; untitled
    // params (no group) always show.
    const sections = builtin.length ? [{ title: 'Display', params: builtin }] : [];
    for (const prm of authored) {
        const last = sections[sections.length - 1];
        if (last && last.authored && last.title === prm.group) last.params.push(prm);
        else sections.push({ title: prm.group, params: [prm], authored: true });
    }
    return sections.map((sec, i) => (sec.title ? (
        <Box key={`s-${i}`} sx={{ mt: i ? 1 : 0 }}>
            <ButtonBase
                onClick={() => onToggleSection(sec.title)}
                aria-expanded={!folded.includes(sec.title)}
                sx={{
                    width: '100%', justifyContent: 'space-between', borderRadius: 0.5,
                    borderTop: i ? '1px solid rgba(255,255,255,0.12)' : 'none', pt: i ? 0.5 : 0,
                    '&:focus-visible': { outline: '1px solid #fff' },
                }}
            >
                <Typography variant="overline" sx={{ opacity: 0.75, lineHeight: 1.8 }}>{sec.title}</Typography>
                <Typography component="span" sx={{ fontSize: 12, opacity: 0.6 }}>
                    {folded.includes(sec.title) ? '▸' : '▾'}
                </Typography>
            </ButtonBase>
            <Collapse in={!folded.includes(sec.title)}>{sec.params.map(control)}</Collapse>
        </Box>
    ) : (
        <Box key={`s-${i}`} sx={{ mt: i ? 1 : 0 }}>{sec.params.map(control)}</Box>
    )));
}
