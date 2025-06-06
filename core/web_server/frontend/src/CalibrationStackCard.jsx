import React, { useContext } from "react";
import { useNavigate, useParams } from "react-router-dom";

import AutoStackCheckbox from "./AutoStackCheckbox.jsx";

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import { CardActionArea, Checkbox, FormGroup, FormControlLabel } from '@mui/material';
import AstroFlowContext from './AstroFlowContext.jsx';
import { AstroFlowImage } from './AstroFlowImage.jsx';

export default function CalibrationStackCard({ stack_id }) {
    const { calibrationStacks } = useContext(AstroFlowContext);
    const stack = calibrationStacks[stack_id];
    console.log('stack_id', stack_id, 'stack', stack);
    const navigate = useNavigate();

    return (
        <Card>
            <CardActionArea
                onClick={event => navigate(`/calibration_stack/${stack_id}`)}
            >
                <CardContent>
                    {
                        <AstroFlowImage
                            objectType="calibration_stack"
                            objectId={stack_id}
                            size="thumbnail"
                            style={{ height: "100%" }}
                        />
                    }
                    <Typography gutterBottom variant="h5" component="div">
                        Calibration Stack {stack.image_type}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        {stack.camera} {stack.filter}
                    </Typography>
                </CardContent>
            </CardActionArea>
        </Card>
    );
}