import React, { useState, useRef, useEffect, useReducer } from "react";
import './Starmap.css';

export default function ExposureDurationWidget({ totalExposureDuration, selectedExposureDuration, sx, ...props }) {

	const clock_face = () => {
		return (
			<circle
				cx="1" cy="1" r="1"
				style={{
					fill: "#333",
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
					stroke: "white",
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
						stroke: "white",
						strokeWidth: 0.35,
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

	function clamp(number, min, max) {
		return Math.max(min, Math.min(max, number));
	}

	const elements = [];
	let numClockFaces = 0;
	while (totalExposureDuration > 0 || selectedExposureDuration > 0) {
		elements.push(<svg
			xmlns="http://www.w3.org/2000/svg"
			viewBox="0 0 2 2"
			style={{ ...sx, "right": `${numClockFaces * 2 + 0.5}em` }}
			key={`key_${numClockFaces}`}
			{...props}
		>
			{clock_face()}
			{wedge(clamp(totalExposureDuration, 0, 11.999), "red")}
			{wedge(clamp(selectedExposureDuration, 0, 11.999), "#4278f5")}
			{tick_marks()}
		</svg >);

		numClockFaces += 1;
		totalExposureDuration -= 12;
		selectedExposureDuration -= 12;
	}

	return <React.Fragment>{elements}</React.Fragment>;
}