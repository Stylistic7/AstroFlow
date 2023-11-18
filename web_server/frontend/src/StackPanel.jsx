import React, { useState, useRef, useEffect, useReducer } from "react";
import TargetPanel from "./TargetPanel.jsx";
import "./Starmap.css";

export default function StackPanel({ stack_id, stacks, handleClick }) {
	let stack_thumbnail_video_url = `/stack_thumbnail_video/${stack_id}`;
	
	return (
		<div style={{margin: "2px"}}>
			<video width="110" loop autoPlay muted>
			  <source src={stack_thumbnail_video_url} type="video/mp4" />
			  Your browser does not support the video tag.
			</video>
		</div>
	);
}
