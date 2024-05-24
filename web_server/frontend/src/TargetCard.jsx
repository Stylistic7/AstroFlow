import React, { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";

import AutoStackCheckbox from "./AutoStackCheckbox.jsx";

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { CardActionArea, Checkbox, FormGroup, FormControlLabel } from '@mui/material';

export default function TargetCard({ stacks, targets, mosaics, target_id }) {
  const target = targets[target_id];
  const navigate = useNavigate();

  let num_columns = 1;

  if (target.stack_ids.length > 4) {
    num_columns = 3;
  } else if (target.stack_ids.length > 1) {
    num_columns = 2;
  }

  return (
    <Card>
      <CardActionArea
        onClick={event => navigate(`/target/${target_id}`)}
      >
        <CardContent>
          <div style={{
            display: "grid",
            gridTemplateColumns: `repeat(${num_columns}, 1fr)`,
            gridColumnGap: "0.5em",
            gridRowGap: "0.5em",
            marginBottom: "0.5em",
          }}>
            {
              target.stack_ids.map(stack_id => <img
                key={`stack_thumbnail_image_${stack_id}`}
                src={`/stack_thumbnail_image/${stack_id}`}
                style={{ width: "100%" }}
              />)
            }
          </div>
          <Typography gutterBottom variant="h5" component="div">
            Target {target.name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Stacks: {target.stack_ids.map(stack_id =>
              stacks[stack_id].filter
            )}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Image scale: {stacks[target.stack_ids[0]].image_scale.toFixed(2)}"/px
          </Typography>
        </CardContent>
      </CardActionArea>
      <AutoStackCheckbox
        object_type="target"
        object_id={target_id}
      />
    </Card>
  );
}