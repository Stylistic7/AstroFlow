import React, { useState, useRef, useEffect, useReducer } from "react";
import StackThumbnail from "./StackThumbnail.jsx";
import "./Starmap.css";

export default function TargetPanel({ target, stacks, handleClick }) {
	// TODO: Don't hardcode 140px.
	return (
		<tr height="140px">
			{target.stack_ids.map((stack_id) => (
				<td key={stack_id} style={{verticalAlign: "text-top"}}>
					<StackThumbnail
						id={"StackThumbnail_" + stack_id}
						stack={stacks[stack_id]}
						width="100%"
						handleClick={handleClick}
					/>
				</td>
		))}
		</tr>
	);
}
