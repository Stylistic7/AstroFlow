import React, { useState, useRef, useEffect, useReducer } from "react";
import TargetPanel from "./TargetPanel.jsx";
import "./Starmap.css";

export default function StackPanel({ stack_id, stacks, handleClick }) {
	const plotRef = useRef();

	let stack_thumbnail_video_url = `/stack_thumbnail_video/${stack_id}`;
	
	const fetchHtml = () => {
	  fetch('/get_plot')
	  .then((response) => {
	    return response.text();
	  })
	  .then((htmlString) => {
	  	if (plotRef.current) {
	  		if (plotRef.current.children[0]) {
				  plotRef.current.removeChild(plotRef.current.children[0]);
				}

	  		const fragment = document.createRange().createContextualFragment(htmlString);
    		plotRef.current.append(fragment);
    	}
	  });
	}

	fetchHtml();

	return (
		<div style={{margin: "2px"}}>
			<video width="30%" loop autoPlay muted style={{float: "left"}}>
			  <source src={stack_thumbnail_video_url} type="video/mp4" />
			  Your browser does not support the video tag.
			</video>
			<div ref={plotRef} width="30%" height="300px" style={{float: "left"}}></div>
		</div>
	);
}
