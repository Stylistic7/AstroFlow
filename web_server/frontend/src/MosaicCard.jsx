import * as React from 'react';
import { useNavigate, useParams } from "react-router-dom";

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';

import MosaicOutline from "./MosaicOutline.jsx";

// TODO: Don't show MosaicCards for "mosaics" with just one panel?
export default function MosaicCard({stacks, targets, mosaics, mosaic_id}) {
  const mosaic = mosaics[mosaic_id];
  const navigate = useNavigate();

  // TODO: Figure out how to show just the "best" channel in the Card.
  return (
    <Card>
      <CardActionArea
        onClick={event => navigate(`/mosaic/${mosaic_id}`)}
      >
        <div id="mosaicPanel"
          style={{
            display: "block",
            margin: "1em",
            position: "relative",
        }}>
          <img
            src={`/mosaic_preview_image/${mosaic_id}`}
            style={{
              height: "100%",
              maxWidth: "100%",
              display: "block",
              marginLeft: "auto",
              marginRight: "auto",
            }}
          />
          <MosaicOutline
            stacks={stacks}
            targets={targets}
            mosaics={mosaics}
            mosaic_id={mosaic_id}
            style={{
              display: "block",
              position: "absolute",
              left: 0,
              top: 0,
              width: "100%",
              height: "100%",
              marginLeft: "auto",
              marginRight: "auto",
              maxWidth: "auto",
              maxHeight: "auto",
            }}
          />
        </div>
        <CardContent>
          <Typography gutterBottom variant="h5" component="div">
            Mosaic {mosaic.name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Image scale:&nbsp;
            {stacks[targets[mosaic.target_ids[0]].stack_ids[0]].image_scale.toFixed(2)}"/px
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Resolution:&nbsp;
            {stacks[targets[mosaic.target_ids[0]].stack_ids[0]].width_pixels}&nbsp;x&nbsp;
            {stacks[targets[mosaic.target_ids[0]].stack_ids[0]].height_pixels}&nbsp;pixels
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}