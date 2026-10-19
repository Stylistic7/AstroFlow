/**
 * Toolbar-embedded HDR controls: an HDR on/off icon toggle plus the display
 * white — how far above the interface's white the canvas may go — once HDR
 * is engaged. Lives in the same toolbar row as the polygon / panel-numbers
 * toggles so no UI ever overlaps the image pixels.
 *
 * The display white is in the finishes' own output units (sRGB-encoded,
 * like everything a finish returns and the canvas shows): 1.0 is SDR white,
 * the brightness of every other pixel on screen, and a value such as 1.35
 * is where the canvas clips, which the browser maps onto the screen's
 * extended range — two stops of headroom. Every finish's curve ends there
 * instead of at 1.0 (its `white()`, finishes/stdlib.py), so the brightest
 * nebulosity gets the headroom and the rest of the picture is untouched.
 *
 * No shipped web API reports the screen's headroom — `(dynamic-range: high)`
 * is a yes/no — so the slider is the control. `useMeasuredHeadroom` follows
 * the screen instead wherever Chrome's flagged ScreenDetailed headroom is
 * exposed.
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
import { srgbDecode, srgbEncode } from './tile_viewer/oklch.js';


const HDR_ONBOARDING_KEY = 'astroflow.hdr_onboarding.seen.v1';
// The slider's reach: 2.0 encoded is about 5.5× SDR white, the most a
// bright extended region can use before a panel's limiter pulls it back.
const MAX_DISPLAY_WHITE = 2.0;


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
                    The display-white slider that appears sets how far
                    above the interface's white the brightest nebulosity
                    may reach (1.35 is two stops). Everything a finish
                    already shows below white stays exactly as it is.
                    Toggle HDR off any time to return to standard SDR
                    rendering.
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
    displayWhite, setDisplayWhite,
}) {
    const [onboardingOpen, setOnboardingOpen] = useState(false);
    useMeasuredHeadroom(hdrCapable && hdrEnabled, setDisplayWhite);

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
                        Display white
                    </Typography>
                    <Slider
                        size="small"
                        value={displayWhite}
                        onChange={(_, v) => setDisplayWhite(v)}
                        min={1} max={MAX_DISPLAY_WHITE} step={0.05}
                        valueLabelDisplay="auto"
                        valueLabelFormat={(v) => `${v.toFixed(2)} (${srgbDecode(v).toFixed(1)}× SDR)`}
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


// Chrome's ScreenDetailed headroom, behind "Experimental Web Platform
// features" (proposed for shipping, 2025-06): `hdrHeadroom` / `headroom` in
// stops, the legacy `highDynamicRangeHeadroom` a linear ratio.
const HEADROOM_PROPS = ['hdrHeadroom', 'headroom', 'highDynamicRangeHeadroom'];

/**
 * Follow the screen's measured headroom where the browser exposes it: the
 * display white tracks the screen, brightness changes included, and the
 * slider shows what was measured. Nothing happens on a browser without the
 * property — the check is on the interface, so the window-management
 * permission is only ever asked for when there is something to read.
 */
function useMeasuredHeadroom(enabled, setDisplayWhite) {
    useEffect(() => {
        const proto = window.ScreenDetailed?.prototype;
        const prop = proto && HEADROOM_PROPS.find((p) => p in proto);
        if (!enabled || !prop || typeof window.getScreenDetails !== 'function') return undefined;
        let details = null;
        const read = () => {
            const v = details.currentScreen[prop];
            const ratio = prop === 'highDynamicRangeHeadroom' ? v : 2 ** v;
            if (ratio >= 1) setDisplayWhite(Math.min(MAX_DISPLAY_WHITE, srgbEncode(ratio)));
        };
        window.getScreenDetails().then((d) => {
            details = d;
            read();
            d.addEventListener('currentscreenchange', read);
        }).catch(() => {});
        return () => details?.removeEventListener('currentscreenchange', read);
    }, [enabled, setDisplayWhite]);
}
