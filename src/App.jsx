import "./App.css"

import { useState, useEffect, useRef } from "react";
import Engine from "./Engine";

export default function App() {
    let [ready, setReady] = useState(0); // 0=checking, 1=setup, 2=listening, 3=error

    const engine = useRef(new Engine(
        () => setReady(2), // onSuccess
        () => setReady(1), // onCreate
        () => setReady(3), // onError
    ));

    return (
        <>
            <div className="topbar">&nbsp;</div>
            {(()=>{
                switch (ready) {
                    case 0:
                        return <div>Checking</div>
                    case 1:
                        return <div>Setting up...</div>
                    case 2:
                        return <div>Listening</div>
                    case 3:
                    default:
                        return <div>Something went wrong.</div>
                }
            })()}
        </>
    )
}

