import { useContext } from "react";
import { useNavigate } from "react-router-dom";

import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import { CardActionArea } from '@mui/material';
import AstroFlowContext from './AstroFlowContext.jsx';
import { AstroFlowImage } from './AstroFlowImage.jsx';

export default function CalibrationStackCard({ stack_id }) {
    const { calibrationStacks } = useContext(AstroFlowContext);
    const stack = calibrationStacks[stack_id];
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