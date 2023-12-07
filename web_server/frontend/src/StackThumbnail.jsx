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

  let stack_thumbnail_image_url = `/stack_thumbnail_image/${stack.id}`;
  return (
  	<div
  	  style={{
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
			    handleClick(stack.id);
			    event.stopPropagation();
			  }}
	    />
	    <div
	      style={{
	      	position: "absolute",
	      	right: "15px",
	      	top: "100px",
	      	fontSize: "12px"
	      }}
	    >
	      <span style={{color: "#aaa"}}>
	      	{stack.image_scale.toFixed(2)}
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