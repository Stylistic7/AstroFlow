import React, { useState } from "react";

import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

import './Starmap.css';

export default function Starmap({constellation_polylines, stacks, targets, mosaics}) {
	const makePolyline = (class_name, polyline, key) => {
		let points_string = "";
		for (const point of polyline) {
		  points_string += (180 + point[0]) + " ";
		  points_string += (90 - point[1]) + " ";
		}

		return (
			<polyline
			  key={key}
				className={class_name}
				points={points_string}
			>
			</polyline>
		);
  };

	return (
		<TransformWrapper>
			{({ zoomIn, zoomOut, resetTransform, ...rest }) => (
        <React.Fragment>
		      <TransformComponent>
		      	<div id="starmap_container">
							<img id="starmap_img" src="/static/images/GAIA_EDR3.png" />
							<span id="starmap_copyright">&copy; ESA/Gaia/DPAC; CC BY-SA 3.0 IGO. Acknowledgement: A. Moitinho.</span>
							<svg id="constellations_svg" viewBox="0 0 360 180">
								{constellation_polylines.map((polyline, index) => 
									makePolyline("constellation_polyline", polyline, index))}
							</svg>
							<svg id="starmap_svg" viewBox="0 0 360 180">
								{Object.values(targets).map((target, index) => 
									makePolyline(
										"starmap_polygon", target.footprint_galactic, index))}
							</svg>
						</div>
					</TransformComponent>
				</React.Fragment>
      )}
    </TransformWrapper>
	);
}