import React, { useState, useRef, useEffect, useReducer } from "react";
import { useNavigate, useParams } from "react-router-dom";

import LayersIcon from '@mui/icons-material/Layers';
import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import DeleteIcon from '@mui/icons-material/Delete';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';

import { trackPromise, usePromiseTracker } from "react-promise-tracker";
import ClipLoader from "react-spinners/ClipLoader";

import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';

import HDRImage from "./HDRImage.jsx";
import TargetPanel from "./TargetPanel.jsx";
import StackThumbnail from "./StackThumbnail.jsx";
import "./Starmap.css";

const override: CSSProperties = {
    display: "block",
    margin: "0 auto",
    borderColor: "red",
};

export default function RunPanel({ run, stacks, flatStackIds }) {
    console.log('flatStackIds', flatStackIds);

    return (
        <div>
            <div><span>Run: {run.content_hash}</span></div>
            <div><span>Available flat stacks:</span></div>
            <ul>
                {flatStackIds.map(id =>
                    <li>{stacks[id].content_hash}: {stacks[id].num_frames} {stacks[id].type} frames</li>)}
            </ul>
        </div>
    );
}
