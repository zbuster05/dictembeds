import "./App.css"
import {ipcRenderer} from "electron";

export default function App() {
    console.log(ipcRenderer);
    return (
        <div className="topbar">&nbsp;</div>
    )
}

