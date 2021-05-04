import {ipcRenderer} from "electron";

export default class Engine {
    constructor(onSuccess=()=>{}, onCreate=()=>{}, onError=()=>{}) {
        ipcRenderer.send('pyenv.check')
        this.status = 'pending';

        this.onSuccess = onSuccess;
        this.onError = onError;
        this.onCreate = onCreate;

        this.setIPCCallbacks();
    }

    setIPCCallbacks() {
        ipcRenderer.on('pyenv.check__reply', (_, args) => (args==="success") ?
            this.startRuntime():this.createRuntime()
        );

        ipcRenderer.on('pyenv.setup__reply', (_, args) => (args==="success") ? 
            this.startRuntime():this.runtimeError()
        );

        ipcRenderer.on('pyserver.start__reply', (_, args) => (args==="success") ?
            this.notifySuccss():this.runtimeError()
        );
    }

    createRuntime() {
        this.status = "creating";
        ipcRenderer.send('pyenv.setup');
        this.onCreate();
    }

    startRuntime() {
        this.status = "starting"
        ipcRenderer.send('pyserver.start')
    }

    runtimeError() {
        this.status = "error"
        this.onError();
    }

    notifySuccss() {
        this.status = "success"
        this.onSuccess();
    }
}

