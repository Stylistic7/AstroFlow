import React, { useState, useRef, useEffect, useReducer } from "react";
import './Starmap.css';

export default function StackThumbnail({stack, width, zoomToElement}) {
	const color_for_filter = (filter) => {
	  let rgb = {
	    'L': [51, 51, 51],
	    'R': [230, 0, 0],
	    'G': [0, 204, 0],
	    'B': [0, 0, 204],
	    'H': [159, 43, 104],
	    'S': [140, 0, 0],
	    'O': [0, 204, 204],
	  }[filter];

	  return 'rgb(' + rgb.join(', ') + ')'
	};

  let stack_thumbnail_image_url = `/stack_thumbnail_image/${stack.id}`;
  return (
  	<div
  	  style={{
  	  	float: "left",
  	  	width: width,
  	  	padding: "0.5px",
  	  	pointerEvents: "none",
  	  	position: "relative",
  	  }}
  	>
	    <img
	    	id={stack.id}
	    	src={stack_thumbnail_image_url}
	    	style={{
	    		pointerEvents: "auto",
	    		width: "95%",
	    		position: "absolute",
	    		top: 0,
	    		left: 0,
	    	}}
	    	onClick={(event) => {
			    setTimeout(() => {zoomToElement(stack.id)}, 100);
			    event.stopPropagation();
			  }}
	    />
	    <svg
	    	xmlns="http://www.w3.org/2000/svg"
	    	style={{
	    		position: "absolute",
	    		top: "-1px",
	    		right: "1px",
	    		width: "2px",
	    		height: "2px"
	    	}}
	    >
	    	<rect
	    		fill="red"
	    		width="2"
	    		height="2"
	    	/>
	    </svg>
	   </div>
  );
}