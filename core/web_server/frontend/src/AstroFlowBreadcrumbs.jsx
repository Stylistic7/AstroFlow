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
import ColorLensIcon from '@mui/icons-material/ColorLens';
import Link from '@mui/material/Link';
import PanoramaHorizontalIcon from '@mui/icons-material/PanoramaHorizontal';
import RuleIcon from '@mui/icons-material/Rule';

import AstroFlowContext from './AstroFlowContext.jsx';
import Chevron from './Chevron.jsx';
import QuarantineToggle from './QuarantineToggle.jsx';
import ShoppingCartMenu from './ShoppingCartMenu.jsx';
import SuggestionBadge from './suggestions/SuggestionBadge.jsx';
import { useFinishes } from './useVersions.js';
import { filterLabel } from './filters.js';

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * A breadcrumb crumb that opens a dropdown of selectable values.
 *
 * Renders an icon + the currently-active value + a chevron. Clicking
 * the chip opens an MUI Menu listing all options; unavailable ones
 * (not built yet) are dimmed and disabled. Keeps the breadcrumb
 * compact even when there are many options.
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

    const { pipeline, finish } = params;

    const mosaic = mosaics[params.mosaic_id];
    const target = targets[params.target_id];
    const stack = stacks[params.stack_id];
    const calStack = calibrationStacks[params.calibration_stack_id];

    // TODO: Don't hardcode the font color; use the theme.
    const fontColor = "#aaa";

    // Breadcrumb-link URLs intentionally drop the viewer's trailing
    // (pipeline, finish) segments — those describe image-viewer state
    // and shouldn't ride along into deeper crumbs.
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
        cal_stack_url += `/calibration_stack/${params.calibration_stack_id}`;
    }

    // The DEEPEST object the URL names is the one being viewed, and its
    // finish list drives the pill crumbs: a calibration stack under a
    // stack, a stack under a target, a target under a mosaic, the mosaic
    // itself. Every one of them renders an image and resolves by its own
    // (type, id).
    let viewerObjectType = null;
    let viewerObjectId = null;
    let viewerPathPrefix = "";
    if (params.calibration_stack_id) {
        viewerObjectType = "calibration_stack";
        viewerObjectId = params.calibration_stack_id;
        viewerPathPrefix = cal_stack_url;
    } else if (params.stack_id) {
        viewerObjectType = "stack";
        viewerObjectId = params.stack_id;
        viewerPathPrefix = stack_url;
    } else if (params.target_id) {
        viewerObjectType = "target";
        viewerObjectId = params.target_id;
        viewerPathPrefix = target_url;
    } else if (params.mosaic_id) {
        viewerObjectType = "mosaic";
        viewerObjectId = params.mosaic_id;
        viewerPathPrefix = mosaic_url;
    }

    const finishesData = useFinishes(
        viewerObjectType, viewerObjectId, computeGraphStatus,
    ) || { pipelines: [] };  // null until the first fetch lands
    // The pipeline pill offers what this OBJECT has: the pipelines whose
    // trace produced a finish for it (the same list the finish pill and the
    // viewer read), not the registry of everything the cart could queue.
    const implementations = finishesData.pipelines.map((p) => p.name);

    // Finishes the current pipeline lists, in preference order. One that is
    // traced but not built yet (`absent`) stays in the menu, dimmed.
    const currentFinishes = finishesData.pipelines.find(
        (p) => p.name === pipeline,
    )?.finishes || [];
    const finishNames = currentFinishes.map((f) => f.name);
    // On a stack page under a target the pill also offers the SIBLING
    // stacks' roles: each stack has exactly one filter, so picking another
    // channel means navigating to a different stack_id.
    const siblingRoles = viewerObjectType === "stack" && target
        ? target.stack_ids.map((sid) => stacks[sid]?.role).filter(Boolean)
        : [];
    const finishOptions = [...new Set([...finishNames, ...siblingRoles])];
    const finishExists = (f) =>
        currentFinishes.find((x) => x.name === f)?.status !== 'absent';

    // Keep the current finish IF the new pipeline lists it; else land on
    // that pipeline's first (preferred) finish, so a pipeline switch never
    // names a finish the target pipeline never built.
    const goToPipeline = (nextPipeline) => {
        const next = finishesData.pipelines.find((p) => p.name === nextPipeline);
        const names = (next?.finishes || []).map((f) => f.name);
        const f = names.includes(finish) ? finish : names[0];
        if (!f) return;
        navigate(`${viewerPathPrefix}/${encodeURIComponent(nextPipeline)}/${encodeURIComponent(f)}`);
    };
    const goToFinish = (f) => {
        if (finishNames.includes(f)) {
            navigate(`${viewerPathPrefix}/${encodeURIComponent(pipeline)}/${encodeURIComponent(f)}`);
            return;
        }
        const siblingId = target?.stack_ids?.find((sid) => stacks[sid]?.role === f);
        if (!siblingId) return;
        const prefix = params.mosaic_id
            ? `/mosaic/${params.mosaic_id}/target/${params.target_id}/stack/${siblingId}`
            : `/target/${params.target_id}/stack/${siblingId}`;
        navigate(`${prefix}/${encodeURIComponent(pipeline)}/${encodeURIComponent(f)}`);
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
                {/* Pipeline pill — switches which pipeline's finish is
                    displayed, carrying the current finish across when the
                    next pipeline lists it. */}
                {pipeline && (
                    <PillCrumb
                        icon={<AccountTreeIcon sx={{ marginRight: '0.5em' }} />}
                        options={implementations}
                        active={pipeline}
                        exists={() => true}
                        onPick={goToPipeline}
                        fontColor={fontColor}
                        activeColor={theme.palette.text.primary}
                    />
                )}
                {/* Finish pill — the display recipe the viewer composes
                    live. Single-channel finishes are named by filter role
                    and labelled like filters; composites show as-is. */}
                {finish && (
                    <PillCrumb
                        icon={<ColorLensIcon sx={{ marginRight: '0.5em' }} />}
                        options={finishOptions}
                        active={finish}
                        exists={finishExists}
                        onPick={goToFinish}
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
