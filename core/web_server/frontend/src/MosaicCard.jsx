import React, { useContext } from "react";
import { useNavigate } from "react-router-dom";

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';

import AstroFlowContext from './AstroFlowContext.jsx';
import PreviewImage from './PreviewImage.jsx';

// TODO: Don't show MosaicCards for "mosaics" with just one panel?
export default function MosaicCard({ mosaic_id }) {
  const { stacks, targets, mosaics } = useContext(AstroFlowContext);

  const mosaic = mosaics[mosaic_id];
  const navigate = useNavigate();

  return (
    <Card>
      <CardActionArea
        onClick={() => navigate(`/mosaic/${mosaic_id}`)}
      >
        <div id="mosaicPanel"
          style={{
            display: "block",
            margin: "1em",
            position: "relative",
          }}>
          {/* The mosaic's DEFAULT finish, baked server-side; renders
              nothing until it has been built. */}
          <PreviewImage
            src={`/thumbnail/preview/mosaic/${mosaic_id}`}
            alt="canonical preview"
            style={{
              height: "100%",
              maxWidth: "100%",
              display: "block",
              marginLeft: "auto",
              marginRight: "auto",
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
