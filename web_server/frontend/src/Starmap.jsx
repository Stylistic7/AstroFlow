import React, { useState, useRef, useEffect, useReducer } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';
import LayersIcon from '@mui/icons-material/Layers';
import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import MosaicPanel from "./MosaicPanel.jsx";
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
		width: 120,
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

	// TODO: Split the popup content into a "Popup" component.
	// TODO: Split CSS up into one file for each component?
	const popupContent = (zoomToElement) => {
		if (!mosaic) {
			return;
		}

		// TODO implement?
		const handleClick = () => {
			console.log('handleClick');
		};

		// TODO: Font sizes are a mess right now. Find some way to gracefully select
		// them and pass them through the hierarchy.
		const fontSize = '2.5px';
		const fontColor = 'black';

		return (
	    <div role="presentation" onClick={handleClick}>
	      <Breadcrumbs
	      	aria-label="breadcrumb"
	      	fontSize={fontSize}
	      	sx={{
	      		color: 'black', ml: 0.15, mt: 0.2, mb: 0
	      	}}>
	        <Link
	          underline="hover"
	          sx={{ display: 'flex', alignItems: 'center', color: fontColor }}
	          fontSize={fontSize}
	          color="inherit"
	          href="/"
	        >
	          <AutoAwesomeMosaicIcon sx={{ ml: 0.1, mr: 0.1 }} fontSize={fontSize} />
	          Mosaic
	        </Link>
	        <Typography
	          sx={{ display: 'flex', alignItems: 'center', color: fontColor }}
	          fontSize={fontSize}
	          color="text.primary"
	        >
	          <LayersIcon sx={{ ml: 0.1, mr: 0.1 }} fontSize={fontSize} />
	          Stack
	        </Typography>
	      </Breadcrumbs>
	      <MosaicPanel
					mosaic={mosaic}
					mosaics={mosaics}
					targets={targets}
					stacks={stacks}
					zoomToElement={zoomToElement}
				/>
	    </div>
	  );
	}

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
								<g
									id="g"
									ref={popupRef}
									x={popupPosition.x}
									y={popupPosition.y}
									width={popupPosition.width}
									height={popupPosition.height}
									visibility={popupVisbility}
									transform={`translate(${popupPosition.x}, ${popupPosition.y})`}
								>
									<rect
									  id="rect_b"
										x="0"
										y="0"
										width={popupPosition.width}
										height={popupPosition.height}
										rx="2"
										style={{ fill: "#999" }}
									/>
									<foreignObject
										id="foreignObject"
										x="0" 
										y="0"
										width={popupPosition.width}
										height={popupPosition.height}
										style={{pointerEvents: "none"}}
									>
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
