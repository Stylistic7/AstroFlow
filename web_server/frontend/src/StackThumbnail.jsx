import React, { useState, useRef, useEffect, useReducer } from "react";
import useVideoFrames from "./useVideoFrames.ts";
import ExposureDurationWidget from "./ExposureDurationWidget.jsx";
import './Starmap.css';

export default function StackThumbnail({ stacks, stack_id, width, handleClick }) {
	const stack = stacks[stack_id];

	const color_for_filter = (filter) => {
		let rgb = {
			'L': [255, 255, 255],
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

	const video_ref = useRef();

	const [frameCount, setFrameCount] = useState(0);

	const [sortedLightFrames, setSortedLightFrames] = React.useState([]);
	const [selectedSubframes, setSelectedSubframes] = React.useState([]);
	const [exposureDuration, setExposureDuration] = React.useState({});

	const fetchHtml = (stack_id) => {
		console.log('fetchHtml');

		fetch(`/get_sorted_light_frames/${stack_id}`)
			.then((response) => {
				return response.json();
			})
			.then((j) => {
				console.log('get_sorted_light_frames', j);
				setSortedLightFrames(j)
			});

		fetch(`/get_selected_subframes/${stack_id}`)
			.then((response) => {
				return response.json();
			})
			.then((j) => {
				console.log('get_selected_subframes', j);
				setSelectedSubframes(j);
			});

		fetch(`/get_exposure_duration/${stack_id}`)
			.then((response) => {
				return response.json();
			})
			.then((j) => {
				console.log('get_exposure_duration', j);
				setExposureDuration(j);
			});
	}

	useEffect(() => {
		fetchHtml(stack_id);
	}, []);

	const [video, setVideo] = useVideoFrames(frameCount => {
		setFrameCount(frameCount);
	});

	useEffect(() => {
		setVideo(video_ref.current);
	}, [video_ref]);

	let light_frames = [];
	for (let [key, run] of Object.entries(stack.runs)) {
		light_frames += run.frames;
	}

	const currentLightFrame = sortedLightFrames[frameCount];
	const currentFrameIsSelected = selectedSubframes.indexOf(currentLightFrame) != -1;

	// TODO: Display the progress bar as red during frames with too few stars.
	return (
		<div className="stack_thumbnail"
			style={{
				pointerEvents: "none",
				position: "relative",
			}}
		>
			{show_image && <img
				id={stack_id}
				src={`/stack_thumbnail_image/${stack_id}`}
				style={{
					pointerEvents: "auto",
					width: "100%",
					top: 0,
					left: 0,
				}}
				onClick={handleClick}
			/>}

			{show_video && <video
				loop
				autoPlay
				muted
				ref={video_ref}
				style={{
					pointerEvents: "auto",
					width: "100%",
					top: 0,
					left: 0,
				}}
				onClick={handleClick}
			>
				<source src={`/stack_thumbnail_video/${stack_id}`} type="video/mp4" />
				Your browser does not support the video tag.
			</video>
			}

			<div
				style={{
					position: "absolute",
					left: "1em",
					top: "1em",
					fontSize: "14px"
				}}
				onClick={handleClick}
			>
				<span style={{ color: "#aaa" }}>
					{stack.camera}, {stack.image_scale.toFixed(2)}"/px,
				</span>
				&nbsp;
				<span style={{
					color: color_for_filter(stack.filter)
				}}>
					{stack.filter}
				</span>
			</div>

			<ExposureDurationWidget
				key={`exposure_duration_${stack_id}`}
				totalExposureDuration={exposureDuration['total_exposure_duration'] / 3600.0}
				selectedExposureDuration={exposureDuration['selected_exposure_duration'] / 3600.0}
				stack={stack}
				sx={{
					position: "absolute",
					top: "0.3em",
					right: "0.5em",
					width: "2em",
				}}
			/>

			<div style={{
				position: "absolute",
				bottom: "0em",
				left: "0",
				width: "100%",
			}}
			>
				<div style={{
					margin: "5px",
					backgroundColor: "#222"
				}}>
					<div style={{
						height: "0.75em",
						width: (100.0 * (frameCount % light_frames.length) / light_frames.length) + "%",
						backgroundColor: currentFrameIsSelected ? "#ccc" : "#f00"
					}}></div>
				</div>
			</div>
		</div>
	);
}