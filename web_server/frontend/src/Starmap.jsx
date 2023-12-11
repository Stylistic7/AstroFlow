import React, { useState, useRef, useEffect, useReducer, forwardRef } from "react";
import { useNavigate } from "react-router-dom";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import Popup from "./Popup.jsx";
import Polyline from "./Polyline.jsx";
import "./Starmap.css";

export default function Starmap({
	constellation_polylines,
	stacks,
	targets,
	mosaics,
}) {
	const navigate = useNavigate();

	const popupRef = useRef(null);

	// TODO: These are really the mosaic id, not the mosaic object itself.
	const [mosaic, setMosaic] = React.useState(undefined);

	const handleClick = (event) => {
		// navigate("/mosaic/" + mosaic.id)}
		console.log('x ' + event.clientX + ', y ' + event.clientY);

		let elements = document.elementsFromPoint(event.clientX, event.clientY);
		let matching_items = [];
	  elements.forEach((elt, i) => {
	    // output.textContent += elt.localName;
	    // if (i < elements.length - 1) {
	    //   output.textContent += " < ";
	    // }
	    console.log('clicked object ', elt.getAttribute("data"));

	   	const data = elt.getAttribute("data");
	    if (data) {
	    	matching_items.push(data);
	    }
	  
	  });

	  navigate(`/disambiguation/${matching_items.join(";")}`);
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
							<svg id="constellations_svg" viewBox="-180 -90 360 180">
								<g transform="scale(1,-1)">
									{constellation_polylines.map((polyline, index) =>
										<Polyline
											className="constellation_polyline"
											polyline={polyline}
											key={index}
										/>
									)}
								</g>
							</svg>
							<svg
								id="starmap_svg"
								viewBox="-180 -90 360 180"
								style={{pointerEvents: "auto"}}
							>
								<g transform="scale(1,-1)">
									{Object.values(targets).map((target) =>
										<Polyline
											className="target_polygon"
											data={['target', target.id]}
											polyline={target.footprint_galactic}
											key={target.id}
											handleClick={event => handleClick(event)}
										/>
									)}
									{Object.values(mosaics).map((mosaic) =>
										<Polyline
											className="mosaic_polygon"
											data={['mosaic', mosaic.id]}
											polyline={mosaic.footprint_galactic}
											key={mosaic.id}
											handleClick={event => handleClick(event)}
										/>
									)}
								</g>
							</svg>
						</div>
					</TransformComponent>
				</React.Fragment>
			)}
		</TransformWrapper>
	);
}
