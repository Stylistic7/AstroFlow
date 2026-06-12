/**
 * Toolbar-embedded HDR controls: an HDR on/off icon toggle plus an
 * inline brightness slider that appears once HDR is engaged. Lives in
 * the same toolbar row as the polygon / panel-numbers toggles so no
 * UI ever overlaps the image pixels.
 *
 * The onboarding dialog fires the first time a user enables HDR (per
 * localStorage flag). HDR mode dims surrounding macOS windows under
 * EDR compositing — worth a heads-up so it doesn't read as a bug.
 */
import { useEffect, useState } from 'react';
import Box from '@mui/material/Box';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import Button from '@mui/material/Button';
import HdrOnIcon from '@mui/icons-material/HdrOn';

import ToggleIconButton from './ToggleIconButton';


const HDR_ONBOARDING_KEY = 'astroflow.hdr_onboarding.seen.v1';


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


export default function HdrControls({
    hdrCapable,
    hdrEnabled, setHdrEnabled,
    peakBrightness, setPeakBrightness,
}) {
    const [onboardingOpen, setOnboardingOpen] = useState(false);

    if (!hdrCapable) return null;

    const handleToggle = () => {
        if (!hdrEnabled && !localStorage.getItem(HDR_ONBOARDING_KEY)) {
            setOnboardingOpen(true);
            return;
        }
        setHdrEnabled(!hdrEnabled);
    };

    const acceptHdr = () => {
        localStorage.setItem(HDR_ONBOARDING_KEY, '1');
        setOnboardingOpen(false);
        setHdrEnabled(true);
    };

    return (
        <>
            <ToggleIconButton
                name="HDR"
                icon={<HdrOnIcon />}
                checked={hdrEnabled}
                handleChange={handleToggle}
            />
            {hdrEnabled && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 180 }}>
                    <Typography variant="caption" sx={{ opacity: 0.7, whiteSpace: 'nowrap' }}>
                        Brightness
                    </Typography>
                    <Slider
                        size="small"
                        value={peakBrightness}
                        onChange={(_, v) => setPeakBrightness(v)}
                        min={1} max={8} step={0.1}
                        valueLabelDisplay="auto"
                        valueLabelFormat={(v) => `${v.toFixed(1)}×`}
                        sx={{ width: 110 }}
                    />
                </Box>
            )}
            <HdrOnboardingDialog
                open={onboardingOpen}
                onAccept={acceptHdr}
                onDecline={() => setOnboardingOpen(false)}
            />
        </>
    );
}


/**
 * Returns true when the display reports HDR-capable via
 * `(dynamic-range: high)` matchMedia. Whether the canvas actually
 * uplifts to HDR is a separate story handled inside TileViewer via
 * the hidden-HLG-video EDR trick — that doesn't need a WebGL feature
 * probe.
 */
export function useHdrCapability() {
    const [capable] = useState(() => {
        if (typeof window === 'undefined' || !window.matchMedia) return false;
        try {
            return window.matchMedia('(dynamic-range: high)').matches;
        } catch {
            return false;
        }
    });
    return capable;
}
