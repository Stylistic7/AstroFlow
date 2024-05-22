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

import ImageGallery from "react-image-gallery";
import "react-image-gallery/styles/css/image-gallery.css";

const override: CSSProperties = {
    display: "block",
    margin: "0 auto",
    borderColor: "red",
};

export default function RunPanel({ stack_id, stacks, flat_stack_ids }) {
    const { promiseInProgress } = usePromiseTracker();

    const set_settings = (stack_id, settings) => {
        console.log('set_settings', stack_id, JSON.stringify(settings));
        trackPromise(
            fetch(`/set_stack_settings/${stack_id}`, {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            })
                .then(r => r.json())
                .then(j => console.log(j))
        );
    };


    console.log('flat_stack_ids', flat_stack_ids);
    let stack = stacks[stack_id];

    const images = flat_stack_ids.map(flat_stack_id => {
        return {
            original: `/stack_thumbnail_images/${stack_id}/${flat_stack_id}`
        }
    });
    console.log('images', images)


    return (
        <div>
            <div><ImageGallery items={images} /></div>
            <div><span>Stack {stack.frames.length} frames, content_hash: {stack.content_hash} flat {stack.dependencies['flat']}</span></div>
            <div><span>Available flat stacks for camera {stack.camera} and filter {stack.filter}:</span></div>
            <ul>
                {flat_stack_ids.map(id =>
                    <li><a onClick={() => set_settings(stack_id, { 'flat': id })}>{stacks[id].content_hash}</a>:&nbsp;
                        {stacks[id].frames.length}&nbsp;
                        {stacks[id].image_type} frames&nbsp;
                        {stacks[id].oldest_date}&nbsp;
                        {stacks[id].newest_date}</li>)}
            </ul>
        </div >
    );
}
