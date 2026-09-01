import React, { useContext, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';

import AstroFlowContext from './AstroFlowContext.jsx';
import PreviewImage from './PreviewImage.jsx';

// TODO: Don't show MosaicCards for "mosaics" with just one panel?
export default function MosaicCard({ mosaic_id }) {
  const { stacks, targets, mosaics } = useContext(AstroFlowContext);
  const [previewAbsent, setPreviewAbsent] = useState(false);

  const mosaic = mosaics[mosaic_id];
  const navigate = useNavigate();

  const get_filters = () => {
    let filters = new Set();
    for (let target_id of mosaic.target_ids) {
      for (let stack_id of targets[target_id].stack_ids) {
        filters.add(stacks[stack_id].filter);
      }
    }
    return Array.from(filters).join(", ");
  }

  // TODO: Display the mosaic image and outline using the same structures
  // used in the MosaicPanel component.
  //
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
          {/* Canonical colour preview (server-rendered Hubble/RGB from the
              linear scene formats); onError falls back to the classic
              per-filter thumbnail (covers never-built images and the deploy
              window before the backend restart). */}
          {previewAbsent ? (
          <img
            src={`/thumbnail/mosaic/${mosaic_id}/${get_filters()[0]}`}
            style={{
              height: "100%",
              maxWidth: "100%",
              display: "block",
              marginLeft: "auto",
              marginRight: "auto",
            }}
          />
          ) : (
          <PreviewImage
            src={`/thumbnail/preview/mosaic/${mosaic_id}`}
            alt="canonical preview"
            onAbsent={() => setPreviewAbsent(true)}
            style={{
              height: "100%",
              maxWidth: "100%",
              display: "block",
              marginLeft: "auto",
              marginRight: "auto",
            }}
          />
          )}
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
          <Typography variant="body2" color="text.secondary">
            Color Channels:&nbsp;
            {get_filters()}
          </Typography>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}