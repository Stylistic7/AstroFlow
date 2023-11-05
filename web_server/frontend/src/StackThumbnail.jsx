import React, { useState, useRef, useEffect, useReducer } from "react";

import './Starmap.css';

export default function StackThumbnail({stack, zoomToElement}) {
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
  console.log('stack_thumbnail_image_url', stack_thumbnail_image_url);
  return (
  	<img
      id={stack.id}
    	key={stack.id}
    	src={stack_thumbnail_image_url}
    	width="15px"
    	onClick={(event) => {
				setTimeout(() => {zoomToElement(stack.id)}, 100);
				event.stopPropagation();
			}}
    />
  );
}