import { useContext, useEffect, useState } from 'react';
import { useParams, useLocation, useNavigate } from "react-router-dom";
import { useTheme } from '@mui/material/styles';

import Box from '@mui/material/Box';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import LayersIcon from '@mui/icons-material/Layers';
import SquareFootIcon from '@mui/icons-material/SquareFoot';
import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import BuildIcon from '@mui/icons-material/Build';
import HighQualityIcon from '@mui/icons-material/HighQuality';
import ColorLensIcon from '@mui/icons-material/ColorLens';
import Link from '@mui/material/Link';
import PanoramaHorizontalIcon from '@mui/icons-material/PanoramaHorizontal';
import RuleIcon from '@mui/icons-material/Rule';

import AstroFlowContext from './AstroFlowContext.jsx';
import QuarantineToggle from './QuarantineToggle.jsx';
import ShoppingCartMenu from './ShoppingCartMenu.jsx';
import SuggestionBadge from './suggestions/SuggestionBadge.jsx';

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}


const QUALITY_ORDER = ["PostProcessed", "Master", "DownscaledMaster", "Subframe"];

/**
 * A breadcrumb crumb that's also a row of selectable pills.
 *
 * Renders an icon + a row of values. The currently-active value is
 * highlighted; values that exist on disk are clickable; values with no
 * corresponding version are dimmed. Matches the typography / padding of
 * the sibling Link crumbs so it reads as part of the breadcrumb trail.
 */
function PillCrumb({ icon, options, active, exists, onPick, fontColor, activeColor }) {
    return (
        <Box sx={{
            display: 'flex',
            alignItems: 'center',
            marginLeft: '0.5em',
            marginRight: '0.5em',
            gap: '0.25em',
        }}>
            {icon}
            {options.map((opt, i) => {
                const isActive = opt === active;
                const isAvailable = exists(opt);
                return (
                    <Box
                        key={opt}
                        component={isAvailable ? 'a' : 'span'}
                        onClick={isAvailable ? (e) => { e.preventDefault(); onPick(opt); } : undefined}
                        href={isAvailable ? '#' : undefined}
                        sx={{
                            color: isActive ? activeColor : fontColor,
                            fontWeight: isActive ? 600 : 400,
                            opacity: isAvailable ? 1 : 0.35,
                            cursor: isAvailable && !isActive ? 'pointer' : 'default',
                            textDecoration: 'none',
                            px: '0.4em',
                            py: '0.1em',
                            borderRadius: '4px',
                            '&:hover': isAvailable && !isActive
                                ? { textDecoration: 'underline' }
                                : {},
                        }}
                    >
                        {opt}
                    </Box>
                );
            })}
        </Box>
    );
}


export default function AstroFlowBreadcrumbs() {
    const { calibrationStacks, stacks, targets, mosaics, computeGraphStatus } = useContext(AstroFlowContext);

    const params = useParams();
    const location = useLocation();
    const navigate = useNavigate();
    const theme = useTheme();

    // Fetch the (implementation × quality × filter) matrix when the URL
    // has a mosaic_id, so the pill crumbs can render with correct
    // enabled/disabled state. Re-fetch on compute-graph-status SSE so the
    // pills light up live as versions finish rendering.
    const [versionsData, setVersionsData] = useState({ filters: [], versions: [] });
    useEffect(() => {
        if (!params.mosaic_id) return;
        fetch(`/api/mosaic_versions/${params.mosaic_id}`)
            .then((res) => res.ok ? res.json() : null)
            .then((data) => data && setVersionsData(data))
            .catch((e) => console.warn(e));
    }, [params.mosaic_id, computeGraphStatus]);

    const { implementation, quality, filter } = params;
    const mosaicVersions = versionsData.versions;

    // Derive the options for each pill row from the versions matrix.
    const implementations = [...new Set(mosaicVersions.map((v) => v.implementation))];
    const qualities = QUALITY_ORDER.filter((q) =>
        mosaicVersions.some(
            (v) => v.quality === q && v.implementation === implementation && v.exists
        )
    );
    const filters = versionsData.filters;

    const versionExists = (impl, q, f) => mosaicVersions.some(
        (v) => v.implementation === impl && v.quality === q && v.filter === f && v.exists
    );

    const goToVersion = (impl, q, f) =>
        navigate(`/mosaic/${params.mosaic_id}/${impl}/${q}/${f}`);

    const mosaic = mosaics[params.mosaic_id];
    const target = targets[params.target_id];
    const stack = stacks[params.stack_id];
    const calStack = calibrationStacks[params.calibration_stack_id];

    // TODO: Don't hardcode the font color; use the theme.
    const fontColor = "#aaa";

    // The mosaic-detail-viewer URL includes (implementation, quality, filter)
    // segments between the mosaic_id and any nested target/stack drilldown,
    // so the breadcrumb back-stack preserves that context.
    const version_path = (params.implementation && params.quality && params.filter)
        ? `/${params.implementation}/${params.quality}/${params.filter}`
        : "";

    let mosaic_url = "";
    let viewer_url = "";
    let target_url = "";
    let stack_url = "";
    let cal_stack_url = "";
    if (params.mosaic_id) {
        mosaic_url = `/mosaic/${params.mosaic_id}`;
        viewer_url = `${mosaic_url}${version_path}`;
        target_url = viewer_url;
        stack_url = viewer_url;
        cal_stack_url = viewer_url;
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
                {implementation && implementations.length > 0 && (
                    <PillCrumb
                        icon={<BuildIcon sx={{ marginRight: '0.5em' }} />}
                        options={implementations}
                        active={implementation}
                        exists={(impl) => versionExists(impl, quality, filter)}
                        onPick={(impl) => goToVersion(impl, quality, filter)}
                        fontColor={fontColor}
                        activeColor={theme.palette.text.primary}
                    />
                )}
                {quality && qualities.length > 0 && (
                    <PillCrumb
                        icon={<HighQualityIcon sx={{ marginRight: '0.5em' }} />}
                        options={qualities}
                        active={quality}
                        exists={(q) => versionExists(implementation, q, filter)}
                        onPick={(q) => goToVersion(implementation, q, filter)}
                        fontColor={fontColor}
                        activeColor={theme.palette.text.primary}
                    />
                )}
                {filter && filters.length > 0 && (
                    <PillCrumb
                        icon={<ColorLensIcon sx={{ marginRight: '0.5em' }} />}
                        options={filters}
                        active={filter}
                        exists={(f) => versionExists(implementation, quality, f)}
                        onPick={(f) => goToVersion(implementation, quality, f)}
                        fontColor={fontColor}
                        activeColor={theme.palette.text.primary}
                    />
                )}
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