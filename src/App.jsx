import "./App.css"

import {ipcRenderer} from "electron";
import { useState, useEffect } from "react";

export default function App() {
    let [ready, setReady] = useState(0); // 0=checking, 1=ready, 2=setup, 3=listening, 4=error

    useEffect(()=>{
        ipcRenderer.send('pyenv.check')
        ipcRenderer.on('pyenv.check__reply', (_, args)=> setReady((args==="success") ? 1 : 2));
        ipcRenderer.on('pyenv.setup__reply', (_, args)=>(args==="success")?setReady(1):setReady(4));
        ipcRenderer.on('pyserver.start__reply', (_, args)=>(args==="success")?setReady(3):setReady(4));
    }, []);

    useEffect(()=>{
        if (ready === 1)
            ipcRenderer.send('pyserver.start')
        else if (ready === 2)
            ipcRenderer.send('pyenv.setup')
    }, [ready]);


    return (
        <>
            <div className="topbar">&nbsp;</div>
            {(()=>{
                switch (ready) {
                    case 0:
                        return <div>Checking</div>
                    case 1:
                        return <div>Ready</div>
                    case 2:
                        return <div>Setting up...</div>
                    case 3:
                        return <div>Listening</div>
                    case 4:
                    default:
                        return <div>Something went wrong.</div>
                }
            })()}
        </>
    )
}

