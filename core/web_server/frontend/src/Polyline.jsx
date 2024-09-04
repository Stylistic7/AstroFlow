import React, { useState, useRef, useEffect, useReducer } from "react";

export default function Polyline({ className, data, polyline, style, handleClick }) {
	let points_string = "";

	for (const point of polyline) {
		let px = point[0];
		let py = point[1];

		points_string += px + " ";
		points_string += py + " ";
	}

	return (
		<polyline
			className={className}
			data={data}
			points={points_string}
			style={style}
			onClick={(event) => { handleClick && handleClick(event) }}
		/>
	);
}

