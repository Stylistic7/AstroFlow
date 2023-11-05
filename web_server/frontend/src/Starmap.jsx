import React, { useState, useRef, useEffect, useReducer } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import Grid from '@mui/material/Unstable_Grid2';
import TargetPanel from "./TargetPanel.jsx";
import "./Starmap.css";

export default function Starmap({
	constellation_polylines,
	stacks,
	targets,
	mosaics,
}) {
	const popupRef = useRef(null);
	const [popupVisbility, setPopupVisibility] = React.useState("hidden");
	const [popupPosition, setPopupPosition] = React.useState({
		x: 30,
		y: 30,
		width: 110,
		height: 100,
	});
	const [mosaic, setMosaic] = React.useState(undefined);

	const makePolyline = (
		class_name,
		polyline,
		key,
		setTransform,
		zoomToElement,
	) => {
		let points_string = "";
		let centroid_x = 0;
		let centroid_y = 0;
		for (const point of polyline) {
			let px = 180 + point[0];
			let py = 90 - point[1];

			points_string += px + " ";
			points_string += py + " ";

			centroid_x += px;
			centroid_y += py;
		}

		centroid_x /= polyline.length;
		centroid_y /= polyline.length;

		return (
			<polyline
				id={key}
				key={key}
				className={class_name}
				points={points_string}
				onClick={(event) => {
					setMosaic(key);
					setPopupPosition({
						...popupPosition,
						x: centroid_x - popupPosition.width - 7,
						y: centroid_y - popupPosition.height / 2,
					});
					setPopupVisibility("visible");
					setTimeout(() => {
						zoomToElement(popupRef.current);
					}, 100);
					event.stopPropagation();
				}}
			></polyline>
		);
	};

	const popupContent = (zoomToElement) => {
		if (!mosaic) {
			return;
		}

		return (
			<div>
				{mosaics[mosaic].target_ids.map((target_id) => (
					<TargetPanel
						target={targets[target_id]}
						stacks={stacks}
						zoomToElement={zoomToElement}
					/>
				))}
			</div>
		);
	};

	// TODO: Figure out why some divs (like those containing text) become huge.
	// Use proper hierarchy for MUI elements.
	return (
		<TransformWrapper maxScale={100} wheel={{ smoothStep: 0.01 }}>
			{({ setTransform, zoomToElement, ...rest }) => (
				<React.Fragment>
					<TransformComponent>
						<div id="starmap_container">
							<img id="starmap_img" src="/static/images/GAIA_EDR3.png" />
							<span id="starmap_copyright">
								&copy; ESA/Gaia/DPAC; CC BY-SA 3.0 IGO. Acknowledgement: A.
								Moitinho.
							</span>
							<svg id="constellations_svg" viewBox="0 0 360 180">
								{constellation_polylines.map((polyline, index) =>
									makePolyline("constellation_polyline", polyline, index),
								)}
							</svg>
							<svg
								id="starmap_svg"
								viewBox="-20 -20 380 200"
								style={{ zIndex: 10 }}
								onClick={() => setPopupVisibility("hidden")}
							>
								{Object.values(targets).map((target) =>
									makePolyline(
										"target_polygon",
										target.footprint_galactic,
										target.id,
										setTransform,
										zoomToElement,
									),
								)}
								{Object.values(mosaics).map((mosaic) =>
									makePolyline(
										"mosaic_polygon",
										mosaic.footprint_galactic,
										mosaic.id,
										setTransform,
										zoomToElement,
									),
								)}
								<g
									ref={popupRef}
									x={popupPosition.x}
									y={popupPosition.y}
									width={popupPosition.width}
									height={popupPosition.height}
									visibility={popupVisbility}
									transform={`translate(${popupPosition.x}, ${popupPosition.y})`}
								>
									<rect
										x="0"
										y="0"
										width={popupPosition.width}
										height={popupPosition.height}
										rx="2"
										style={{ fill: "none" }}
									/>
									<rect
										x="0"
										y="0"
										width={popupPosition.width - 4}
										height={popupPosition.height - 4}
										rx="2"
										style={{ fill: "#999" }}
									/>
									<foreignObject x="5" y="0" width="100" height="100">
										{popupContent(zoomToElement)}
									</foreignObject>
								</g>
							</svg>
						</div>
					</TransformComponent>
				</React.Fragment>
			)}
		</TransformWrapper>
	);
}
