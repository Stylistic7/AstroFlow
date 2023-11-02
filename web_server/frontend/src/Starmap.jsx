import React, { useState, useRef, useEffect, useReducer } from "react";

import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

import './Starmap.css';

export default function Starmap({constellation_polylines, stacks, targets, mosaics}) {
	const popupRef = useRef(null);
	const [popupVisbility, setPopupVisibility] = React.useState("hidden");
	const [popupPosition, setPopupPosition] = React.useState({x: 30, y: 30, width: 50, height: 50});
	const [target, setTarget] = React.useState(undefined);

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
					setTarget(key);
					setPopupPosition({
						...popupPosition,
						x: centroid_x - popupPosition.width - 7,
						y: centroid_y - popupPosition.height / 2
				  });
					setPopupVisibility("visible");
					setTimeout(() => {zoomToElement(popupRef.current)}, 100);
					event.stopPropagation();
				}}
			>
			</polyline>
		);
  };

  const stackThumbnailImage = (stack_id, index, zoomToElement) => {
  	let stack = stacks[stack_id];
    let stack_thumbnail_image_url = `/stack_thumbnail_image/${stack.id}`;
    let key = target.id + "_" + stack_id;
    console.log('stack_thumbnail_image_url', stack_thumbnail_image_url);
    return (
      <image
        id={key}
      	key={key}
      	href={stack_thumbnail_image_url}
      	x={popupPosition.x + 4 + index * 7}
      	y={popupPosition.y + 6}
      	width="6"
      	onClick={(event) => {
					setTimeout(() => {zoomToElement(key)}, 100);
					event.stopPropagation();
				}}
      />
    );
  };

  const popupContent = (zoomToElement) => {
  	if (!target) {
  		return;
  	}

  	return (
  		<g>
  			<text
	  		  x={popupPosition.x + 4}
					y={popupPosition.y + 5}
					fontSize="2"
	  		>
	  			{targets[target].name}
	  		</text>
	  		{ targets[target].stack_ids.map(
	  			(stack_id, index) => stackThumbnailImage(stack_id, index, zoomToElement)) }
  		</g>
  	);
  };

  // TODO: Use the <g> transform property.
	return (
		<TransformWrapper maxScale={100} wheel={{smoothStep: 0.01}}>
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
							<svg id="starmap_svg" viewBox="-20 -20 380 200" style={{zIndex: 10}}>
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
										style={{fill: '#999'}}
									/>
									{popupContent(zoomToElement)}
								</g>
							</svg>
						</div>
					</TransformComponent>
				</React.Fragment>
      )}
    </TransformWrapper>
	);
}