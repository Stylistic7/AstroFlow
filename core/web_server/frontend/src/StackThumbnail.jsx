import { useState, useRef, useEffect, useContext } from "react";
import useVideoFrames from "./useVideoFrames.ts";
import ExposureDurationWidget from "./ExposureDurationWidget.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';

export default function StackThumbnail({ stack_id, width, handleClick }) {
	const { stacks, invalidated } = useContext(AstroFlowContext);
	const stack = stacks[stack_id];
	const isStale = (invalidated || []).some(
		([type, id]) => type === "stack" && id === stack_id
	);

	const color_for_filter = (filter) => {
		const key = filter?.[0]?.toUpperCase?.();
		const rgb = {
			'L': [255, 255, 255],
			'R': [230, 0, 0],
			'G': [0, 204, 0],
			'B': [0, 0, 204],
			'H': [159, 43, 104],
			'S': [140, 0, 0],
			'O': [0, 204, 204],
		}[key] || [128, 128, 128];  // fallback to gray if unknown

		return `rgb(${rgb.join(', ')})`;
	};

	const show_image = false;
	const show_video = true;
	const video_fps = 10;

	const video_ref = useRef();

	const [frameCount, setFrameCount] = useState(0);

	const currentFrameIsSelected = true;

	const [video, setVideo] = useVideoFrames(mediaTime => {
		setFrameCount(mediaTime * video_fps);
	});

	useEffect(() => {
		setVideo(video_ref.current);
	}, [video_ref]);

	if (!stack) return null;

	const numFrames = (stack.filenames || []).length || 1;

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
				src={`/thumbnail/stack/${stack_id}/none`}
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
				<source src={`/video/stack/${stack_id}`} type="video/mp4" />
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
					{stack.camera}, {stack.image_scale.toFixed(2)}"/px, {(stack.rotation_angle * 180 / Math.PI).toFixed(2)}&deg;
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
				totalExposureDuration={stack.integration / 3600.0}
				selectedExposureDuration={stack.integration / 3600.0}
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
						width: (100.0 * (frameCount % numFrames) / numFrames) + "%",
						backgroundColor: currentFrameIsSelected ? "#ccc" : "#f00"
					}}></div>
				</div>
			</div>

			<div style={{
				position: "absolute",
				bottom: "5em",
				left: "0",
				width: "100%",
			}}
			>
				<span>Frame count {frameCount}</span>
			</div>

			{isStale && (
				<div style={{
					position: "absolute", top: "0.5em", right: "0.5em",
					background: "rgba(255, 100, 100, 0.85)",
					color: "white", padding: "1px 6px",
					borderRadius: 4, fontSize: 11,
					pointerEvents: "none",
				}}>
					Reprocessing...
				</div>
			)}

		</div>
	);
}