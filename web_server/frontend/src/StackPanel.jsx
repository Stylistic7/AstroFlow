import React, { useState, useRef, useEffect, useReducer } from "react";
import { useNavigate, useParams } from "react-router-dom";

import LayersIcon from '@mui/icons-material/Layers';
import AutoAwesomeMosaicIcon from '@mui/icons-material/AutoAwesomeMosaic';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';

import Breadcrumbs from '@mui/material/Breadcrumbs';
import Link from '@mui/material/Link';
import Typography from '@mui/material/Typography';

import TargetPanel from "./TargetPanel.jsx";
import "./Starmap.css";

// TODO: The StackPanel should contain the following information:
// 
// - status
//   - {not stacked, stacked, post-processed}
// 
// - stack health
//   - sanity check results
// 
// - visualization of data quality (e.g. star count, clouds, etc.)
//   - video
//   - plot
// 
// - results
//   - stacked master
//   - post-processed
//
// and the following actions:
//
// - quarantine or delete the entire stack
// - quarantine or delete just the poor images in the stack
// - select flat frames
// - queue the stack for stacking
// - configure stacking options and post-processing scripts
export default function StackPanel({ stacks, targets, mosaics }) {
	const params = useParams();
	const stack_id = params.stack_id;
	const stack = stacks[stack_id];

	const plotRef = useRef();

	// TODO: Don't hardcode the font color; use the theme.
	const fontColor = "#aaa";
	
	const fetchHtml = (stack_id) => {
	  fetch(`/get_num_stars_histogram_html/${stack_id}`)
	  .then((response) => {
	    return response.text();
	  })
	  .then((htmlString) => {
	  	if (plotRef.current) {
	  		if (plotRef.current.children[0]) {
				  plotRef.current.removeChild(plotRef.current.children[0]);
				}

	  		const fragment = document.createRange().createContextualFragment(htmlString);
    		plotRef.current.append(fragment);
    	}
	  });
	}

	fetchHtml(stack_id);

	let stack_thumbnail_video_url = `/stack_thumbnail_video/${stack_id}`;

	return (
		<>
			<Breadcrumbs aria-label="breadcrumb">
	      <Link
	      	href="/"
	        underline="hover"
	        sx={{
	        	display: 'flex',
	        	alignItems: 'center',
	        	color: fontColor,
	        	marginLeft: '0.5em',
	        	marginRight: '0.5em',
	        }}
	        style={{pointerEvents: "auto"}}
	        color="inherit"
	      >
	        <AutoAwesomeMosaicIcon sx={{ marginRight: '0.5em' }}/>
	        Starmap
	      </Link>
	      <Typography color="text.primary" margin="0.5em">
	      	{stack.target} Stack
	      </Typography>
	    </Breadcrumbs>

			<div style={{margin: "2px"}}>
				<video width="30%" loop autoPlay muted style={{float: "left"}}>
				  <source src={stack_thumbnail_video_url} type="video/mp4" />
				  Your browser does not support the video tag.
				</video>
				<div ref={plotRef} width="30%" height="300px" style={{float: "left"}}></div>
				<div style={{
					position: "relative",
					background: "black",
					width: "240px",
					height: "240px"
				}}>
					<video poster="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQAQAAAAA3iMLMAAAAAXNSR0IArs4c6QAAAA5JREFUeNpj+P+fgRQEAP1OH+HeyHWXAAAAAElFTkSuQmCC" src="data:video/mp4;base64,AAAAHGZ0eXBpc29tAAACAGlzb21pc28ybXA0MQAAAAhmcmVlAAAAvG1kYXQAAAAfTgEFGkdWStxcTEM/lO/FETzRQ6gD7gAA7gIAA3EYgAAAAEgoAa8iNjAkszOL+e58c//cEe//0TT//scp1n/381P/RWP/zOW4QtxorfVogeh8nQDbQAAAAwAQMCcWUTAAAAMAAAMAAAMA84AAAAAVAgHQAyu+KT35E7gAADFgAAADABLQAAAAEgIB4AiS76MTkNbgAAF3AAAPSAAAABICAeAEn8+hBOTXYAADUgAAHRAAAAPibW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAAKcAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAw10cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAAKcAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAABAAAAAQAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAACnAAAAAAABAAAAAAKFbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAABdwAAAD6BVxAAAAAAAMWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABDb3JlIE1lZGlhIFZpZGVvAAAAAixtaW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAHsc3RibAAAARxzdHNkAAAAAAAAAAEAAAEMaHZjMQAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAQABAASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAAHVodmNDAQIgAAAAsAAAAAAAPPAA/P36+gAACwOgAAEAGEABDAH//wIgAAADALAAAAMAAAMAPBXAkKEAAQAmQgEBAiAAAAMAsAAAAwAAAwA8oBQgQcCTDLYgV7kWVYC1CRAJAICiAAEACUQBwChkuNBTJAAAAApmaWVsAQAAAAATY29scm5jbHgACQAQAAkAAAAAEHBhc3AAAAABAAAAAQAAABRidHJ0AAAAAAAALPwAACz8AAAAKHN0dHMAAAAAAAAAAwAAAAIAAAPoAAAAAQAAAAEAAAABAAAD6AAAABRzdHNzAAAAAAAAAAEAAAABAAAAEHNkdHAAAAAAIBAQGAAAAChjdHRzAAAAAAAAAAMAAAABAAAAAAAAAAEAAAfQAAAAAgAAAAAAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAQAAAABAAAAJHN0c3oAAAAAAAAAAAAAAAQAAABvAAAAGQAAABYAAAAWAAAAFHN0Y28AAAAAAAAAAQAAACwAAABhdWR0YQAAAFltZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAACxpbHN0AAAAJKl0b28AAAAcZGF0YQAAAAEAAAAATGF2ZjYwLjMuMTAw"
						muted="muted"
						autoplay="autoplay"
						playsinline=""
						style={{
						  position: "absolute",
						  top: "0",
						  left: "0",
						  width: "100%",
						  height: "100%",
					}}></video>
					<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAeAAAAHgAQMAAABHGizWAAAAAXNSR0IB2cksfwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAZQTFRFAAAAAAAApWe5zwAAAAJ0Uk5TAP9bkSK1AAABqElEQVR4nO3aQXKEIBBA0XY1x/CocFSO4UoStGlAsGRmKln9v7E0PHaUDkGEiIiIiIiIiP4nF0tB3C6vdIlxS39Ld6UdDAZPYXuUlCx5pOHQjwSDwQ/4da43XYuxmqN5CAaDv8DutF7AYPDXOA1YY+7NJQkGgxVr+qWYpvJLHv7WtycYDD4exZINry8lMBg8hbvq993EcDAYfE2XpJd6EXoR/Yrc8pDx5yMYDB6sseV3gI5bq82NLf99zVP53oLB4DsseSPe2+VYi7YNchcYDB7i9BbTXDzvlnLRAhgMnsVStjOCbXzoV+RS341XJhgM7lrqL8XYnh5s1N2PMjAY3OGUt0OEW5nRNVP53oLB4AF2e1FrPorRTny/JMFg8OhdlSo7h/pfrmo/vgQGg+ex5c7d+WhvOBFbtXdLEgwGd8OtkB5cjzuF6icaGAyewfZIvyJjN9ztl5FgMPgB215hyL/NvJ3P0DYwGPwh1jvfLlCdAQwGf47juePh+/O6YDB4Bms67kjfd9VZwtBTMBg8xrEUqg1E2/goPe2TgMFgIiIiIiIiIvrDfgCHetV+sEUfrQAAAABJRU5ErkJggg=="
						alt="Reference"
						style={{
							display: "block",
							position: "absolute",
							top: "0",
							left: "0",
							width: "100%",
							height: "100%",
					}}></img>
				</div>
			</div>
		</>
	);
}
