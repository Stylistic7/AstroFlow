import React, { useState, useRef, useEffect, useReducer } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import Popup from "./Popup.jsx";
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
		width: 120,
		height: 100,
	});
	// TODO: These are really the mosaic id, not the mosaic object itself.
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
					let x = centroid_x;
					if (centroid_x < 180) {
						x = x + 7;
					} else {
						x = x - popupPosition.width - 7
					}

					setMosaic(key);
					setPopupPosition({
						...popupPosition,
						x: x,
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

	// TODO: Handle clicks on Stacks.
	//
	// TODO: Split CSS up into one file for each component?
	//
	// TODO: Figure out why some divs (like those containing text) become huge.
	// Use proper hierarchy for MUI elements.
	return (
		<TransformWrapper maxScale={100} wheel={{ smoothStep: 0.01 }}>
			{({ setTransform, zoomToElement, ...rest }) => (
				<React.Fragment>
					<TransformComponent>
						<div id="starmap_container">
							<img id="starmap_img" src="/static/images/GAIA_EDR3_darkened.jpg" />
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
								viewBox="0 0 360 180"
								style={{pointerEvents: "auto"}}
								onClick={() => {
									console.log('starmap_svg got click');
									setPopupVisibility("hidden");
								}}
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
							</svg>
							<Popup
								mosaic_id={mosaic}
								stack_id={undefined}
								mosaics={mosaics}
								targets={targets}
								stacks={stacks}
								zoomToElement={zoomToElement}
							/>
						</div>
					</TransformComponent>
				</React.Fragment>
			)}
		</TransformWrapper>
	);
}
