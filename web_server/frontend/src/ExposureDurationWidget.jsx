import React, { useState, useRef, useEffect, useReducer } from "react";
import './Starmap.css';

export default function ExposureDurationWidget({ totalExposureDuration, selectedExposureDuration, sx }) {

	const clock_face = () => {
		return (
			<circle
				cx="1" cy="1" r="1"
				style={{
					fill: "#aaa",
				}}
			/>
		);
	}

	const tick_mark = (hour) => {
		const theta = (2 * Math.PI * hour / 12) - (Math.PI / 2);
		const start_x = 1 + 0.75 * Math.cos(theta);
		const start_y = 1 + 0.75 * Math.sin(theta);
		const end_x = 1 + Math.cos(theta);
		const end_y = 1 + Math.sin(theta);

		return (
			<path
				key={hour}
				d={`M ${start_x} ${start_y} L ${end_x} ${end_y}`}
				style={{
					stroke: "black",
					strokeWidth: 0.1,
				}}
				clipPath="url(#insideCircleOnly)" />
		);
	};

	const tick_marks = () => {
		return (
			<>
				<defs>
					<clipPath id="insideCircleOnly">
						<circle cx="1" cy="1" r="1" />
					</clipPath>
				</defs>
				{Array(12).fill().map((_, hour) => tick_mark(hour))}
				<circle
					cx="1" cy="1" r="1"
					style={{
						fill: "none",
						stroke: "black",
						strokeWidth: 0.2,
					}}
					clipPath="url(#insideCircleOnly)"
				/>
			</>
		);
	};

	const wedge = (hour, color) => {
		const hours = 12;
		const theta = (2 * Math.PI * hour / 12) - (Math.PI / 2);
		const large_arc = hour > 6 ? "1" : "0";
		const end_x = 1 + Math.cos(theta);
		const end_y = 1 + Math.sin(theta);

		return (
			<path d={`M 1 0 A 1 1 0 ${large_arc} 1 ${end_x} ${end_y} L 1 1 Z`} fill={color} />
		);
	};

	// TODO: Make this show the actual good and total exposure times.
	return (<svg
		xmlns="http://www.w3.org/2000/svg"
		viewBox="0 0 2 2"
		style={sx}
	>
		{clock_face()}
		{wedge(totalExposureDuration, "red")}
		{wedge(selectedExposureDuration, "blue")}
		{tick_marks()}
	</ svg>);
}