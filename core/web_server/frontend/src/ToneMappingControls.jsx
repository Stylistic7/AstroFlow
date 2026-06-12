/**
 * Floating overlay over the TileViewer with the HDR controls. The
 * auto-stretch itself is not parameterized — it's the same PixInsight
 * STF the server-side `autostretch` runs — so this panel only renders
 * the HDR toggle + brightness multiplier, gated on display capability.
 */
import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Switch from '@mui/material/Switch';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';


const PANEL_SX = {
    position: 'absolute',
    top: 12,
    right: 12,
    zIndex: 10,
    width: 240,
    padding: '12px 14px',
    backgroundColor: 'rgba(20, 20, 22, 0.78)',
    border: '1px solid rgba(255, 255, 255, 0.12)',
    borderRadius: 8,
    color: '#fff',
    backdropFilter: 'blur(6px)',
    pointerEvents: 'auto',
    userSelect: 'none',
};

const SLIDER_SX = {
    color: '#7da9ff',
    '& .MuiSlider-thumb': { width: 12, height: 12 },
    '& .MuiSlider-rail': { opacity: 0.4 },
};

const HDR_ONBOARDING_KEY = 'astroflow.hdr_onboarding.seen.v1';


function LabeledSlider({ label, value, onChange, min, max, step, suffix }) {
    return (
        <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between',
                       alignItems: 'baseline', mb: -0.5 }}>
                <Typography variant="caption" sx={{ opacity: 0.85 }}>
                    {label}
                </Typography>
                <Typography variant="caption" sx={{ opacity: 0.6, fontFeatureSettings: '"tnum"' }}>
                    {suffix ? `${value.toFixed(2)}${suffix}` : value.toFixed(2)}
                </Typography>
            </Box>
            <Slider
                size="small"
                value={value}
                onChange={(_, v) => onChange(v)}
                min={min}
                max={max}
                step={step}
                sx={SLIDER_SX}
            />
        </Box>
    );
}


function HdrOnboardingDialog({ open, onAccept, onDecline }) {
    return (
        <Dialog open={open} maxWidth="sm">
            <DialogTitle>HDR display mode</DialogTitle>
            <DialogContent>
                <Typography variant="body2" gutterBottom>
                    AstroFlow can drive the image canvas above the SDR
                    luminance ceiling on your HDR-capable display, so
                    the brightest nebulosity in your data reads at full
                    XDR brightness instead of being clipped to the white
                    point of every other UI element.
                </Typography>
                <Typography variant="body2" gutterBottom sx={{ mt: 1 }}>
                    <strong>Heads up:</strong> macOS dims surrounding
                    windows the moment any HDR canvas paints. The
                    browser chrome, the rest of AstroFlow, and any other
                    app on the same display will look noticeably darker
                    while HDR mode is engaged. This is the OS
                    compositor — not a bug.
                </Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                    Use the brightness slider that appears to push
                    star cores past 1.0 nits and into extended display
                    luminance. Toggle HDR off any time to return to
                    standard SDR rendering.
                </Typography>
            </DialogContent>
            <DialogActions>
                <Button onClick={onDecline}>Not now</Button>
                <Button onClick={onAccept} variant="contained">Enable HDR</Button>
            </DialogActions>
        </Dialog>
    );
}


export default function ToneMappingControls({
    peakBrightness, setPeakBrightness,
    hdrEnabled, setHdrEnabled,
    hdrCapable,
}) {
    const [onboardingOpen, setOnboardingOpen] = useState(false);

    // No knobs in SDR mode → nothing to show. Keeps the canvas clear.
    if (!hdrCapable) return null;

    const handleHdrToggle = (e) => {
        const next = e.target.checked;
        if (next && !localStorage.getItem(HDR_ONBOARDING_KEY)) {
            setOnboardingOpen(true);
            return;
        }
        setHdrEnabled(next);
    };

    const acceptHdr = () => {
        localStorage.setItem(HDR_ONBOARDING_KEY, '1');
        setOnboardingOpen(false);
        setHdrEnabled(true);
    };

    const declineHdr = () => setOnboardingOpen(false);

    return (
        <>
            <Box sx={PANEL_SX}>
                <Stack spacing={1.25}>
                    <Box sx={{ display: 'flex', alignItems: 'center',
                                justifyContent: 'space-between' }}>
                        <Typography variant="caption" sx={{ opacity: 0.85 }}>
                            HDR display
                        </Typography>
                        <Switch
                            size="small"
                            checked={hdrEnabled}
                            onChange={handleHdrToggle}
                        />
                    </Box>
                    {hdrEnabled && (
                        <LabeledSlider
                            label="HDR brightness"
                            value={peakBrightness}
                            onChange={setPeakBrightness}
                            min={1} max={8} step={0.1}
                            suffix="×"
                        />
                    )}
                </Stack>
            </Box>
            <HdrOnboardingDialog
                open={onboardingOpen}
                onAccept={acceptHdr}
                onDecline={declineHdr}
            />
        </>
    );
}


export function useHdrCapability() {
    const [capable, setCapable] = useState(() => {
        if (typeof window === 'undefined' || !window.matchMedia) return false;
        try {
            return window.matchMedia('(dynamic-range: high)').matches;
        } catch {
            return false;
        }
    });
    useEffect(() => {
        if (typeof window === 'undefined' || !window.matchMedia) return;
        let mq;
        try {
            mq = window.matchMedia('(dynamic-range: high)');
        } catch {
            return;
        }
        const handler = (e) => setCapable(e.matches);
        if (mq.addEventListener) mq.addEventListener('change', handler);
        else mq.addListener(handler);
        return () => {
            if (mq.removeEventListener) mq.removeEventListener('change', handler);
            else mq.removeListener(handler);
        };
    }, []);
    return capable;
}
