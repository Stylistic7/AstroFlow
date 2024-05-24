import React, { useEffect, useContext } from "react";
import DisambiguationPanel from "./DisambiguationPanel.jsx";
import AstroFlowContext from './AstroFlowContext.jsx';

export default function RecentPanel() {
  const { stacks, targets, mosaics } = useContext(AstroFlowContext);
  const [recent, setRecent] = React.useState(undefined);

  const fetchRecent = () => {
    console.log('fetchRecent');

    fetch(`/recent`)
      .then((response) => {
        return response.json();
      })
      .then((j) => {
        let recentString = j.map(s => `target,${s}`).join(";");
        console.log('recentString', recentString);
        setRecent(recentString);
      });
  }

  useEffect(() => {
    fetchRecent();
  }, []);

  return (
    recent !== undefined && <DisambiguationPanel
      stacks={stacks}
      targets={targets}
      mosaics={mosaics}
      items_in={recent}
    />
  );
}


