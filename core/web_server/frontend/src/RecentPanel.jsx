import { useEffect, useState } from "react";
import DisambiguationPanel from "./DisambiguationPanel.jsx";

export default function RecentPanel() {
  const [recent, setRecent] = useState(undefined);

  const fetchRecent = () => {
    fetch('/api/recent')
      .then((response) => response.json())
      .then((j) => {
        const recentString = j.map(s => `${s[0]},${s[1]}`).join(";");
        setRecent(recentString);
      });
  }

  useEffect(() => {
    fetchRecent();
  }, []);

  return (
    recent !== undefined && <DisambiguationPanel
      items_in={recent}
    />
  );
}


