import React, { useState, useRef, useEffect, useReducer } from "react";
import ExposureDurationWidget from "./ExposureDurationWidget.jsx";
import './Starmap.css';

export default function StackThumbnail({stack, width, handleClick}) {
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

	const show_image = false;
	const show_video = true;

  return (
  	<div class="stack_thumbnail"
  	  style={{
  	  	pointerEvents: "none",
  	  	position: "relative",
  	  }}
  	>
	    {show_image && <img
	    	id={stack.id}
	    	src={`/stack_thumbnail_image/${stack.id}`}
	    	style={{
	    		pointerEvents: "auto",
	    		width: "100%",
	    		position: "absolute",
	    		top: 0,
	    		left: 0,
	    	}}
	    	onClick={(event) => {
			    handleClick(stack.id);
			    event.stopPropagation();
			  }}
	    />}
	    {show_video && <video
	    	loop
	    	autoPlay
	    	muted
	    	style={{
	    		pointerEvents: "auto",
	    		position: "absolute",
	    		width: "100%",
	    		top: 0,
	    		left: 0,
	    	}}
	    	>
          <source src={`/stack_thumbnail_video/${stack.id}`} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      }
	    <div
	      style={{
	      	position: "absolute",
	      	right: "4em",
	      	top: "0.5em",
	      	fontSize: "12px"
	      }}
	    >
	      <span style={{color: "#aaa"}}>
	      	{stack.width_pixels} x {stack.height_pixels}
	      </span>
	      &nbsp;
	      <span style={{color: "#aaa"}}>
	      	{stack.image_scale.toFixed(2)}"/px
	      </span>
	      &nbsp;
	      <span style={{
	      	color: color_for_filter(stack.filter)
	      }}>
	      	{stack.filter}
	      </span>
	    </div>
    	<ExposureDurationWidget
    		key={`exposure_duration_${stack.id}`}
    		stack={stack}
    		width="20px"
    	/>
	   </div>
  );
}