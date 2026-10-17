/**
 * A stack's card on the target page: ONE still image and what it is.
 *
 * `src` shows a given image instead (a target's pipeline section passes that
 * pipeline's master, with its `status` for the OUT OF DATE chip); by default
 * `/thumbnail/stack/<id>` serves the master's default finish when the stack
 * has been stacked (`X-Producer: master`, status live / out_of_date), else
 * its best raw subframe by star count (`X-Producer: subframe`, status
 * absent). The chips say which; the rest of the overlay is what the card
 * can state truthfully from `/api/stacks`: filter, subs and hours, newest
 * night. (Until 2026-10-01 this played a per-subframe video that was never
 * generated.)
 */
import { useState, useContext } from "react";
import ExposureDurationWidget from "./ExposureDurationWidget.jsx";
import PreviewImage from "./PreviewImage.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';

const FILTER_COLORS = {
	L: [255, 255, 255], R: [230, 0, 0], G: [0, 204, 0], B: [0, 0, 204],
	H: [159, 43, 104], S: [140, 0, 0], O: [0, 204, 204],
};

function colorForFilter(filter) {
	const rgb = FILTER_COLORS[filter?.[0]?.toUpperCase?.()] || [128, 128, 128];
	return `rgb(${rgb.join(', ')})`;
}

const chipStyle = (background) => ({
	background, color: "white", padding: "1px 6px", borderRadius: 4,
	fontSize: 11, fontWeight: 600, letterSpacing: 0.3, pointerEvents: "none",
});

export default function StackThumbnail({ stack_id, handleClick, children, src, status }) {
	const { stacks, invalidated } = useContext(AstroFlowContext);
	const [served, setServed] = useState(null);   // {status, producer} | null
	const [absent, setAbsent] = useState(false);
	const stack = stacks[stack_id];
	if (!stack) return null;

	const isStale = (invalidated || []).some(
		([type, id]) => type === "stack" && id === stack_id
	);
	const numSubs = (stack.filenames || []).length;
	const hours = (stack.raw_integration_s ?? stack.integration ?? 0) / 3600.0;
	const newest = stack.newest_date ? String(stack.newest_date).slice(0, 10) : null;

	return (
		<div className="stack_thumbnail" style={{ position: "relative", cursor: "pointer" }}
			onClick={handleClick}>
			{absent ? (
				<div style={{
					width: "100%", aspectRatio: "3 / 2", background: "#111",
					display: "flex", alignItems: "center", justifyContent: "center",
					color: "#666", fontSize: 13,
				}}>
					no readable frame yet
				</div>
			) : (
				<PreviewImage
					src={src || `/thumbnail/stack/${stack_id}`}
					alt={`${stack.name} ${stack.filter}`}
					style={{ width: "100%", display: "block" }}
					onLoaded={setServed}
					onAbsent={() => setAbsent(true)}
				/>
			)}

			<div style={{
				position: "absolute", left: "0.5em", bottom: "0.5em",
				display: "flex", alignItems: "center", gap: "0.4em",
				fontSize: 13, textShadow: "0 0 4px black", pointerEvents: "none",
			}}>
				<span style={{ color: colorForFilter(stack.filter), fontWeight: 700 }}>
					{stack.filter}
				</span>
				<span style={{ color: "#ddd" }}>
					{numSubs} subs · {hours.toFixed(1)} h{newest ? ` · ${newest}` : ""}
				</span>
			</div>

			<div style={{
				position: "absolute", top: "0.3em", right: "0.5em",
				display: "flex", flexDirection: "row", alignItems: "center", gap: "0.3em",
				pointerEvents: "auto",
			}} onClick={(e) => e.stopPropagation()}>
				<ExposureDurationWidget
					key={`exposure_duration_${stack_id}`}
					totalExposureDuration={hours}
					stack={stack}
					sx={{ width: "2em" }}
				/>
				{children}
			</div>

			<div style={{
				position: "absolute", top: "0.5em", left: "0.5em",
				display: "flex", gap: "0.3em", alignItems: "center",
			}}>
				{served?.producer === "subframe" && (
					<span style={chipStyle("rgba(60, 60, 60, 0.85)")}>
						unstacked · best of {numSubs} subs
					</span>
				)}
				{status === "out_of_date" && (
					<span style={chipStyle("rgba(237, 108, 2, 0.9)")}>OUT OF DATE</span>
				)}
				{isStale && (
					<span style={chipStyle("rgba(255, 100, 100, 0.85)")}>Reprocessing…</span>
				)}
			</div>
		</div>
	);
}
