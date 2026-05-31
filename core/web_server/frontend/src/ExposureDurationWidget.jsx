export default function ExposureDurationWidget({
	totalExposureDuration,
	wbppEffectiveDuration,
	sx,
	...props
}) {
	// Two-layer clock face:
	//   • Blue wedge sized to `totalExposureDuration` (raw total across
	//     every subframe — the upper bound).
	//   • Green wedge OVERLAID on top, sized to `wbppEffectiveDuration`
	//     (PCL:TotalExposureTime from the master — what WBPP actually
	//     kept after its own rejection pass). Always ≤ total.
	//   • When `wbppEffectiveDuration` is null/undefined (no master yet,
	//     or non-PixInsight stacker), only the blue wedge renders —
	//     the user sees a single-color widget identical to the old
	//     single-value appearance.
	//   • When wbpp === total (nothing rejected), green fully covers
	//     blue and reads as "perfect retention."

	const clock_face = () => (
		<circle
			cx="1" cy="1" r="1"
			style={{ fill: "#333" }}
		/>
	);

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
				style={{ stroke: "white", strokeWidth: 0.1 }}
				clipPath="url(#insideCircleOnly)" />
		);
	};

	const tick_marks = () => (
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

	const wedge = (hour, color) => {
		const theta = (2 * Math.PI * hour / 12) - (Math.PI / 2);
		const large_arc = hour > 6 ? "1" : "0";
		const end_x = 1 + Math.cos(theta);
		const end_y = 1 + Math.sin(theta);

		return (
			<path d={`M 1 0 A 1 1 0 ${large_arc} 1 ${end_x} ${end_y} L 1 1 Z`} fill={color} />
		);
	};

	const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

	// One 12-hour clock face per 12h block of total exposure. The wbpp
	// arc on each face is the portion of *that face's* 12h slice that
	// WBPP kept — so a long-integration stack with mild rejection shows
	// proportionally on every face.
	const elements = [];
	let remainingTotal = totalExposureDuration;
	let remainingWbpp = wbppEffectiveDuration;
	let numClockFaces = 0;
	while (remainingTotal > 0) {
		const totalThisFace = clamp(remainingTotal, 0, 11.999);
		const hasWbpp = remainingWbpp != null && remainingWbpp > 0;
		const wbppThisFace = hasWbpp ? clamp(remainingWbpp, 0, 11.999) : 0;

		elements.push(
			<svg
				xmlns="http://www.w3.org/2000/svg"
				viewBox="0 0 2 2"
				style={{ ...sx, right: `${numClockFaces * 2 + 0.5}em` }}
				key={`key_${numClockFaces}`}
				{...props}
			>
				{clock_face()}
				{wedge(totalThisFace, "#4278f5")}
				{hasWbpp && wedge(wbppThisFace, "#4caf50")}
				{tick_marks()}
			</svg>
		);

		numClockFaces += 1;
		remainingTotal -= 12;
		if (remainingWbpp != null) remainingWbpp -= 12;
	}

	return <>{elements}</>;
}
