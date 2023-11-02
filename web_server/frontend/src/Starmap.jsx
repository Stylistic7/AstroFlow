import React, { useState, useRef, useEffect, useReducer } from "react";

import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

import './Starmap.css';

export default function Starmap({constellation_polylines, stacks, targets, mosaics}) {
	const popupRef = useRef(null);
	const [popupVisbility, setPopupVisibility] = React.useState("hidden");
	const [popupPosition, setPopupPosition] = React.useState({x: 30, y: 30, width: 50, height: 50});

	const makePolyline = (class_name, polyline, key, setTransform, zoomToElement) => {
		let points_string = "";
		let centroid_x = 0;
		let centroid_y = 0;
		for (const point of polyline) {
			let px = (180 + point[0]);
			let py = (90 - point[1]);

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
					setPopupPosition({
						...popupPosition,
						x: centroid_x - popupPosition.width - 7,
						y: centroid_y - popupPosition.height / 2
				  });
					setPopupVisibility("visible");
					setTimeout(() => {zoomToElement(popupRef.current);}, 100);
					event.stopPropagation();
				}}
			>
			</polyline>
		);
  };

	return (
		<TransformWrapper limitToBounds={false}>
			{({ setTransform, zoomToElement, ...rest }) => (
        <React.Fragment>
		      <TransformComponent>
		      	<div id="starmap_container">
							<img
								id="starmap_img"
								src="/static/images/GAIA_EDR3.png" />
							<span id="starmap_copyright">&copy; ESA/Gaia/DPAC; CC BY-SA 3.0 IGO. Acknowledgement: A. Moitinho.</span>
							<svg id="constellations_svg" viewBox="0 0 360 180">
								{constellation_polylines.map((polyline, index) => 
									makePolyline("constellation_polyline", polyline, index))}
							</svg>
							<svg id="starmap_svg" viewBox="-10 -10 370 190" style={{zIndex: 10}}>
								{Object.values(targets).map(target => 
									makePolyline(
										"starmap_polygon", target.footprint_galactic, target.id,
										setTransform, zoomToElement))}
								<g ref={popupRef}
								  x={popupPosition.x}
									y={popupPosition.y}
									width={popupPosition.width}
									height={popupPosition.height}
								  visibility={popupVisbility}
								>
								  <rect
								  	x={popupPosition.x}
										y={popupPosition.y}
										width={popupPosition.width}
										height={popupPosition.height}
										rx="2"
										style={{fill: 'none'}}
								  />
									<rect
									  x={popupPosition.x + 2}
										y={popupPosition.y + 2}
										width={popupPosition.width - 4}
										height={popupPosition.height - 4}
										rx="2"
										style={{fill: 'white'}}
									/>
								</g>
							</svg>
						</div>
					</TransformComponent>
				</React.Fragment>
      )}
    </TransformWrapper>
	);
}