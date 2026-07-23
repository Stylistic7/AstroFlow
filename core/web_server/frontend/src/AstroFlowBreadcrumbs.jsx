import { useContext, useRef, useState } from 'react';
import { useParams, useLocation, useNavigate } from "react-router-dom";
import { useTheme } from '@mui/material/styles';

import Box from '@mui/material/Box';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import LayersIcon from '@mui/icons-material/Layers';
import SquareFootIcon from '@mui/icons-material/SquareFoot';
import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import AccountTreeIcon from '@mui/icons-material/AccountTree';
import PhotoIcon from '@mui/icons-material/Photo';
import ColorLensIcon from '@mui/icons-material/ColorLens';
import Link from '@mui/material/Link';
import PanoramaHorizontalIcon from '@mui/icons-material/PanoramaHorizontal';
import RuleIcon from '@mui/icons-material/Rule';

import AstroFlowContext from './AstroFlowContext.jsx';
import Chevron from './Chevron.jsx';
import QuarantineToggle from './QuarantineToggle.jsx';
import ShoppingCartMenu from './ShoppingCartMenu.jsx';
import SuggestionBadge from './suggestions/SuggestionBadge.jsx';
import {
    useVersionsMatrix,
    usePipelines,
    pipelineOptions,
} from './useVersions.js';
import { defaultDisplayFor } from './urls.js';
import { filterLabel } from './filters.js';

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * A breadcrumb crumb that opens a dropdown of selectable values.
 *
 * Renders an icon + the currently-active value + a chevron. Clicking
 * the chip opens an MUI Menu listing all options; unavailable ones
 * (no version on disk) are dimmed and disabled. Keeps the breadcrumb
 * compact even when there are many options (filters can have 7+).
 */
function PillCrumb({ icon, options, active, exists, onPick, fontColor, activeColor, formatOption }) {
    const [open, setOpen] = useState(false);
    const anchorRef = useRef(null);
    const label = formatOption ? formatOption : (v) => v;

    return (
        <Box sx={{
            display: 'flex',
            alignItems: 'center',
            marginLeft: '0.5em',
            marginRight: '0.5em',
        }}>
            {icon}
            <Box
                ref={anchorRef}
                onClick={() => setOpen((o) => !o)}
                sx={{
                    color: activeColor,
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.25em',
                    px: '0.4em',
                    py: '0.1em',
                    borderRadius: '4px',
                    '&:hover': { backgroundColor: 'rgba(255,255,255,0.06)' },
                }}
            >
                {label(active)}
                <Chevron open={open} size={14} color={fontColor} />
            </Box>
            <Menu
                anchorEl={anchorRef.current}
                open={open}
                onClose={() => setOpen(false)}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
                transformOrigin={{ vertical: 'top', horizontal: 'left' }}
            >
                {options.map((opt) => {
                    const isActive = opt === active;
                    const isAvailable = exists(opt);
                    return (
                        <MenuItem
                            key={opt}
                            selected={isActive}
                            disabled={!isAvailable}
                            onClick={() => {
                                setOpen(false);
                                if (!isActive && isAvailable) onPick(opt);
                            }}
                            sx={{
                                fontWeight: isActive ? 600 : 400,
                                opacity: isAvailable ? 1 : 0.4,
                            }}
                        >
                            {label(opt)}
                        </MenuItem>
                    );
                })}
            </Menu>
        </Box>
    );
}


export default function AstroFlowBreadcrumbs() {
    const { calibrationStacks, stacks, targets, mosaics, computeGraphStatus } = useContext(AstroFlowContext);

    const params = useParams();
    const location = useLocation();
    const navigate = useNavigate();
    const theme = useTheme();

    const { pipeline, display, filter } = params;

    const mosaic = mosaics[params.mosaic_id];
    const target = targets[params.target_id];
    const stack = stacks[params.stack_id];
    const calStack = calibrationStacks[params.calibration_stack_id];

    // TODO: Don't hardcode the font color; use the theme.
    const fontColor = "#aaa";

    // Breadcrumb-link URLs intentionally drop the viewer's trailing
    // (implementation, quality, filter) segments — those describe
    // image-viewer state and shouldn't ride along into deeper crumbs.
    let mosaic_url = "";
    let target_url = "";
    let stack_url = "";
    let cal_stack_url = "";
    if (params.mosaic_id) {
        mosaic_url = `/mosaic/${params.mosaic_id}`;
        target_url = mosaic_url;
        stack_url = mosaic_url;
        cal_stack_url = mosaic_url;
    }
    if (params.target_id) {
        target_url += `/target/${params.target_id}`;
        stack_url += `/target/${params.target_id}`;
        cal_stack_url += `/target/${params.target_id}`;
    }
    if (params.stack_id) {
        stack_url += `/stack/${params.stack_id}`;
        cal_stack_url += `/stack/${params.stack_id}`;
    }
    if (params.calibration_stack_id) {
        cal_stack_url += `/calibration_stack/${params.stack_id}`;
    }

    // Pick the deepest object that the URL identifies — that's the one
    // whose version matrix drives the pill crumbs. Stack viewer pages
    // (even when nested under a mosaic) show stack versions; the bare
    // mosaic viewer page shows mosaic versions. Target / calibration
    // stack pages don't render an image, so they don't get pills.
    let viewerObjectType = null;
    let viewerObjectId = null;
    let viewerPathPrefix = "";
    if (params.stack_id) {
        viewerObjectType = "stack";
        viewerObjectId = params.stack_id;
        viewerPathPrefix = stack_url;
    } else if (params.mosaic_id && pipeline) {
        viewerObjectType = "mosaic";
        viewerObjectId = params.mosaic_id;
        viewerPathPrefix = mosaic_url;
    }
    // Per [[project_url_grammar]] every tap-keyed URL carries the
    // full view triple (pipeline / display / filter). Stack URLs
    // include `:filter` even though it's redundant with `stack.filter`.
    const viewerHasFilter = viewerObjectType !== null;

    const versionsData = useVersionsMatrix(
        viewerObjectType, viewerObjectId, computeGraphStatus,
    ) || { filters: [], pipelines: [] };  // null until the first fetch lands
    const availablePipelines = usePipelines();
    const implementations = pipelineOptions(versionsData.pipelines, availablePipelines);
    const filters = versionsData.filters || [];

    // Displays the current pipeline emits (the tap-point labels:
    // "Mosaic", "Channel", "Post-Processed", "Subframe", ...). One
    // pipeline can produce several taps, so the user picks which to
    // view.
    const currentPipelineData = versionsData.pipelines.find(
        (p) => p.name === pipeline,
    );
    const displays = (currentPipelineData?.taps || []).map((t) => t.display);

    // URLs are pipeline+display-keyed: /<obj>/<id>/<pipeline>/<display>[/<filter>].
    // The pill-cycle callers pass the next pipeline name; we pick a
    // display that THAT pipeline actually emits. Carrying the current
    // display across pipelines lets you land on combos like
    // (PixInsight Downscaled, Subframe) which never produce a file —
    // the dashboard's tap-keyed URL then 404s.
    //
    // Selection order: keep current `display` IF the new pipeline
    // emits it; else fall back to the new pipeline's first tap; else
    // the per-type default.
    const goToPipeline = (nextPipeline, f) => {
        const nextPipelineData = versionsData.pipelines.find(
            (p) => p.name === nextPipeline,
        );
        const validDisplays = (nextPipelineData?.taps || []).map((t) => t.display);
        const d = (display && validDisplays.includes(display)) ? display
            : (validDisplays[0] || defaultDisplayFor(viewerObjectType));
        const segs = [
            viewerPathPrefix,
            encodeURIComponent(nextPipeline),
            encodeURIComponent(d),
        ];
        if (viewerHasFilter && f) segs.push(encodeURIComponent(f));
        navigate(segs.filter(Boolean).join("/"));
    };
    const goToFilter = (f) => {
        if (!pipeline) return;
        const d = display || defaultDisplayFor(viewerObjectType);
        // On a stack page, switching filter means showing a sibling
        // stack — each stack has exactly one filter, so we have to
        // navigate to a different stack_id. Resolve via target.stack_ids
        // when target context exists; otherwise fall through to the
        // generic URL build (which won't actually switch filter, but
        // at least won't 404).
        if (
            viewerObjectType === "stack"
            && params.target_id
            && stack
            && f !== stack.filter
        ) {
            const t = targets[params.target_id];
            const siblingId = t?.stack_ids?.find(
                (sid) => stacks?.[sid]?.filter === f,
            );
            if (siblingId) {
                const prefix = params.mosaic_id
                    ? `/mosaic/${params.mosaic_id}/target/${params.target_id}/stack/${siblingId}`
                    : `/target/${params.target_id}/stack/${siblingId}`;
                navigate(
                    `${prefix}/${encodeURIComponent(pipeline)}/${encodeURIComponent(d)}/${f}`,
                );
                return;
            }
        }
        navigate(
            `${viewerPathPrefix}/${encodeURIComponent(pipeline)}/${encodeURIComponent(d)}/${f}`,
        );
    };
    const goToDisplay = (nextDisplay) => {
        if (!pipeline) return;
        const segs = [
            viewerPathPrefix,
            encodeURIComponent(pipeline),
            encodeURIComponent(nextDisplay),
        ];
        // Filter is always part of the view triple in the URL grammar,
        // even on stack pages where it's intrinsic. Without it we lose
        // the filter context and React Router can't match the route.
        if (viewerHasFilter && filter) segs.push(encodeURIComponent(filter));
        navigate(segs.filter(Boolean).join("/"));
    };

    // Determine the deepest (most specific) object for the quarantine toggle.
    let quarantineObjectType = null;
    let quarantineObjectId = null;
    if (params.calibration_stack_id && calStack) {
        quarantineObjectType = "stack";
        quarantineObjectId = params.calibration_stack_id;
    } else if (params.stack_id && stack) {
        quarantineObjectType = "stack";
        quarantineObjectId = params.stack_id;
    } else if (params.target_id && target) {
        quarantineObjectType = "target";
        quarantineObjectId = params.target_id;
    } else if (params.mosaic_id && mosaic) {
        quarantineObjectType = "mosaic";
        quarantineObjectId = params.mosaic_id;
    }

    return (
        <Box sx={{
            display: 'flex',
            alignItems: 'center',
            backgroundColor: theme.palette.background.default,
            pr: 1,
        }}>
            <Breadcrumbs aria-label="breadcrumb" sx={{
                padding: "12px",
                flex: 1,
            }}>
                <Link
                    href="/"
                    underline="hover"
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        color: fontColor,
                        marginLeft: '0.5em',
                        marginRight: '0.5em',
                    }}
                    style={{ pointerEvents: "auto" }}
                    color="inherit"
                >
                    <PanoramaHorizontalIcon sx={{ marginRight: '0.5em' }} />
                    Starmap
                </Link>
                {params.items && <Link
                    href={location.pathname}
                    underline="hover"
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        color: fontColor,
                        marginLeft: '0.5em',
                        marginRight: '0.5em',
                    }}
                    style={{ pointerEvents: "auto" }}
                    color="inherit"
                >
                    <RuleIcon sx={{ marginRight: '0.5em' }} />
                    Disambiguation
                </Link>}
                {params.mosaic_id && <Link
                    href={mosaic_url}
                    underline="hover"
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        color: fontColor,
                        marginLeft: '0.5em',
                        marginRight: '0.5em',
                    }}
                    style={{ pointerEvents: "auto" }}
                    color="inherit"
                >
                    <AutoAwesomeMosaicIcon sx={{ marginRight: '0.5em' }} />
                    Mosaic "{mosaic.name}"
                </Link>}
                {params.target_id && <Link
                    href={target_url}
                    underline="hover"
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        color: fontColor,
                        marginLeft: '0.5em',
                        marginRight: '0.5em',
                    }}
                    style={{ pointerEvents: "auto" }}
                    color="inherit"
                >
                    <AutoAwesomeIcon sx={{ marginRight: '0.5em' }} />
                    Target "{target.name}"
                </Link>}
                {params.stack_id && <Link
                    href={stack_url}
                    underline="hover"
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        color: fontColor,
                        marginLeft: '0.5em',
                        marginRight: '0.5em',
                    }}
                    style={{ pointerEvents: "auto" }}
                    color="inherit"
                >
                    <LayersIcon sx={{ marginRight: '0.5em' }} />
                    Stack "{stack.name}"
                </Link>}
                {params.calibration_stack_id && <Link
                    href={cal_stack_url}
                    underline="hover"
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        color: fontColor,
                        marginLeft: '0.5em',
                        marginRight: '0.5em',
                    }}
                    style={{ pointerEvents: "auto" }}
                    color="inherit"
                >
                    <SquareFootIcon sx={{ marginRight: '0.5em' }} />
                    {capitalizeFirst(calStack.image_type)} calibration stack
                </Link>}
                {/* Pipeline pill — URL is pipeline-keyed; the pill
                    just switches which pipeline's tap is displayed
                    (carrying the current display label across, falling
                    back to a sensible default for the object type if
                    none was set). */}
                {pipeline && implementations.length > 0 && (
                    <PillCrumb
                        icon={<AccountTreeIcon sx={{ marginRight: '0.5em' }} />}
                        options={implementations}
                        active={pipeline}
                        exists={() => true}
                        onPick={(p) => goToPipeline(p, filter)}
                        fontColor={fontColor}
                        activeColor={theme.palette.text.primary}
                    />
                )}
                {/* Display (tap-point) pill — selects which of the
                    current pipeline's emitted taps ("Mosaic",
                    "Channel", "Post-Processed", "Subframe", ...) the
                    viewer renders. Always rendered when `display` is
                    in the URL so the breadcrumbs mirror the URL's view
                    triple, even when there's only one option. */}
                {display && (
                    <PillCrumb
                        icon={<PhotoIcon sx={{ marginRight: '0.5em' }} />}
                        options={displays}
                        active={display}
                        exists={() => true}
                        onPick={goToDisplay}
                        fontColor={fontColor}
                        activeColor={theme.palette.text.primary}
                    />
                )}
                {filter && filters.length > 0 && (
                    <PillCrumb
                        icon={<ColorLensIcon sx={{ marginRight: '0.5em' }} />}
                        options={filters}
                        active={filter}
                        exists={() => true}
                        onPick={goToFilter}
                        fontColor={fontColor}
                        activeColor={theme.palette.text.primary}
                        formatOption={filterLabel}
                    />
                )}
            </Breadcrumbs>
            {quarantineObjectType && (<>
                <ShoppingCartMenu
                    objectType={quarantineObjectType}
                    objectId={quarantineObjectId}
                />
                <QuarantineToggle
                    objectType={quarantineObjectType}
                    objectId={quarantineObjectId}
                />
                {(quarantineObjectType === "target" || quarantineObjectType === "mosaic") && (
                    <SuggestionBadge
                        objectType={quarantineObjectType}
                        objectId={quarantineObjectId}
                    />
                )}
            </>)}
        </Box>
    );
}