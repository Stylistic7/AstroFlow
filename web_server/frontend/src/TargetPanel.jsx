import React, { useState, useRef, useEffect, useReducer } from "react";
import StackThumbnail from "./StackThumbnail.jsx";
import "./Starmap.css";

export default function TargetPanel({ target, stacks, zoomToElement }) {
	return (
		<div className="row" style={{pointerEvents: "none"}}> 
				{target.stack_ids.map((stack_id) => (
					<StackThumbnail
						id={"StackThumbnail_" + stack_id}
						key={stack_id}
						stack={stacks[stack_id]}
						width={(100 / target.stack_ids.length) + '%'}
						zoomToElement={zoomToElement}
					/>
				))}
		</div>
	);
}
