import React, { useState, useEffect, useContext } from "react";

import { trackPromise, usePromiseTracker } from "react-promise-tracker";
import ClipLoader from "react-spinners/ClipLoader";

import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';
import CheckIcon from '@mui/icons-material/Check';

import HDRImage from "./HDRImage.jsx";
import TargetPanel from "./TargetPanel.jsx";
import StackThumbnail from "./StackThumbnail.jsx";
import "./Starmap.css";
import AstroFlowContext from './AstroFlowContext.jsx';

import ImageGallery from "react-image-gallery";
import "react-image-gallery/styles/css/image-gallery.css";


const override: CSSProperties = {
    display: "block",
    margin: "0 auto",
    borderColor: "red",
};

export default function RunPanel({ stack_id, flat_stack_ids }) {
    const { stacks } = useContext(AstroFlowContext);

    const { promiseInProgress } = usePromiseTracker();
    const [stackSettings, setStackSettings] = useState({});

    useEffect(() => {
        fetch(`/get_stack_dependencies/${stack_id}`)
            .then(r => r.json())
            .then(j => setStackSettings(j));
    }, []);

    const set_settings = (stack_id, settings) => {
        console.log('set_settings', stack_id, JSON.stringify(settings));
        fetch(`/set_stack_settings/${stack_id}`, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
    };

    console.log('flat_stack_ids', flat_stack_ids);
    let stack = stacks[stack_id];

    // TODO: The "thumbnail" here is really 1200 pixels wide. Consider
    // generating some kind of hierarchy of legitimate thumbnails.
    const images = flat_stack_ids.map(flat_stack_id => {
        return {
            flat_stack_id: flat_stack_id,
            original: `/flat_correction_thumbnails/${stack_id}/${flat_stack_id}`,
            thumbnail: `/flat_correction_thumbnails/${stack_id}/${flat_stack_id}`
        }
    });
    console.log('images', images)

    const onSlide = (currentIndex) => {
        console.log('currentIndex', currentIndex,
            'flat_stack_id', images[currentIndex].flat_stack_id);
        set_settings(stack_id, { flat: images[currentIndex].flat_stack_id });
    };

    return (
        <div>
            <div><img key={Date.now()} src={`/stack_thumbnail_image/${stack_id}`} /></div>
            <div><span>Stack {stack.frames.length} frames, content_hash: {stack.content_hash} flat {stack.dependencies['flat']}</span></div>
            <div><span>Available flat stacks for camera {stack.camera} and filter {stack.filter}:</span></div>
            <ul>
                {flat_stack_ids.map(id =>
                    <li>
                        {stackSettings['flat'] == id ? <CheckIcon /> : null}
                        <a style={{ cursor: 'pointer' }} onClick={() => set_settings(stack_id, { 'flat': id })}>{stacks[id].content_hash}</a>:&nbsp;
                        {stacks[id].frames.length}&nbsp;
                        {stacks[id].image_type} frames&nbsp;
                        {stacks[id].oldest_date}&nbsp;
                        {stacks[id].newest_date}
                    </li>)}
            </ul>
        </div >
    );
}
