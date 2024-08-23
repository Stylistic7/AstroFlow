import React, { useState, useRef, useEffect, useReducer } from "react";
import { useParams } from "react-router-dom";

import Typography from '@mui/material/Typography';

import "./Starmap.css";

export default function MultiChannelImage({ stacks, targets, mosaics, mosaic_id }) {
  const mosaic = mosaics[mosaic_id];

  const colorForFilter = (filter) => {
    let rgb = {
      'L': [51, 51, 51],
      'R': [230, 0, 0],
      'G': [0, 204, 0],
      'B': [0, 0, 204],
      'H': [159, 43, 104],
      'S': [140, 0, 0],
      'O': [0, 204, 204],
    }[filter];

    return 'rgb(' + rgb.join(', ') + ')'
  };

  const getFilters = () => {
    let filter_array = [];

    for (let filter of ['H', 'S', 'O', 'L', 'R', 'G', 'B']) {
      let found = false;
      for (let target_id of mosaic.target_ids) {
        for (let stack_id of targets[target_id].stack_ids) {
          if (stacks[stack_id].filter == filter) {
            found = true;
          }
        }
      }

      if (found) {
        filter_array.push(filter);
      }
    }

    return filter_array;

    // const filter_array = Array.from(filters);
    // console.log('got filters', filter_array);
    // return filter_array;
  }

  const filters = getFilters();
  const [filter, setFilter] = useState(filters[0]);

  const imageType = "mosaic_thumbnail_image_for_filter";

  return (
    <div style={{}}>
      <img
        src={`/${imageType}/${mosaic_id}/${filter}`}
        style={{
          maxHeight: "90vh",
          maxWidth: "100%",
          display: "block",
          marginLeft: "auto",
          marginRight: "auto",
        }}
      />
      <div
        style={{
          width: "100%",
          display: "flex",
          justifyContent: "center"
        }}
      >
        {filters.map(f => {
          const color = colorForFilter(f);
          return <div
            style={{
              float: "left",
              margin: "0.5em",
              padding: "0.5em",
              border: f == filter ? `1px solid ${color}` : "",
            }}
          >
            <span
              style={{
                color: color,
                margin: "0.5em"
              }}
              onClick={event => setFilter(f)}
            >
              {f}
            </span>
          </div>
        })}
      </div >
    </div >
  );
}
