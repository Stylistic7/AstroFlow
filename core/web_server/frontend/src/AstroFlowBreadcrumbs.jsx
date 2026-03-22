import React, { useState, useRef, useEffect, useReducer } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import { useContext } from 'react';
import { styled, useTheme } from '@mui/material/styles';

import Box from '@mui/material/Box';
import Breadcrumbs from '@mui/material/Breadcrumbs';
import LayersIcon from '@mui/icons-material/Layers';
import SquareFootIcon from '@mui/icons-material/SquareFoot';
import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import Link from '@mui/material/Link';
import PanoramaHorizontalIcon from '@mui/icons-material/PanoramaHorizontal';
import RuleIcon from '@mui/icons-material/Rule';

import AstroFlowContext from './AstroFlowContext.jsx';
import QuarantineToggle from './QuarantineToggle.jsx';

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}


export default function AstroFlowBreadcrumbs() {
    const { calibrationStacks, stacks, targets, mosaics } = useContext(AstroFlowContext);

    const params = useParams();
    const location = useLocation();
    const theme = useTheme();

    const mosaic = mosaics[params.mosaic_id];
    const target = targets[params.target_id];
    const stack = stacks[params.stack_id];
    const calStack = calibrationStacks[params.calibration_stack_id];

    // TODO: Don't hardcode the font color; use the theme.
    const fontColor = "#aaa";

    let mosaic_url = "";
    let target_url = "";
    let stack_url = "";
    let cal_stack_url = "";
    if (params.mosaic_id) {
        mosaic_url += `/mosaic/${params.mosaic_id}`;
        target_url += `/mosaic/${params.mosaic_id}`;
        stack_url += `/mosaic/${params.mosaic_id}`;
        cal_stack_url += `/mosaic/${params.mosaic_id}`;
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
    let quarantineIsQuarantined = false;
    if (params.calibration_stack_id && calStack) {
        quarantineObjectType = "stack";
        quarantineObjectId = params.calibration_stack_id;
        quarantineIsQuarantined = calStack.is_quarantined || false;
    } else if (params.stack_id && stack) {
        quarantineObjectType = "stack";
        quarantineObjectId = params.stack_id;
        quarantineIsQuarantined = stack.is_quarantined || false;
    } else if (params.target_id && target) {
        quarantineObjectType = "target";
        quarantineObjectId = params.target_id;
        quarantineIsQuarantined = target.is_quarantined || false;
    } else if (params.mosaic_id && mosaic) {
        quarantineObjectType = "mosaic";
        quarantineObjectId = params.mosaic_id;
        quarantineIsQuarantined = mosaic.is_quarantined || false;
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
            </Breadcrumbs>
            {quarantineObjectType && (
                <QuarantineToggle
                    objectType={quarantineObjectType}
                    objectId={quarantineObjectId}
                    isQuarantined={quarantineIsQuarantined}
                />
            )}
        </Box>
    );
}