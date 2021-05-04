import {ipcRenderer} from "electron";

export default class Engine {
    constructor(onSuccess=()=>{}, onCreate=()=>{}, onStop=()=>{}, onError=()=>{}) {
        this.status = 'stopped';

        this.onSuccess = onSuccess;
        this.onCreate = onCreate;
        this.onStop = onStop;
        this.onError = onError;

        this.armIPCCallbacks();
    }

    start() {
        ipcRenderer.send('pyenv.check')
    }

    armIPCCallbacks() {
        ipcRenderer.on('pyenv.check__reply', (_, args) => (args==="success") ?
            this.startRuntime():this.createRuntime()
        );

        ipcRenderer.on('pyenv.setup__reply', (_, args) => (args==="success") ? 
            this.startRuntime():this.runtimeError()
        );

        ipcRenderer.on('pyserver.start__reply', (_, args) => (args==="success") ?
            this.notifySuccss():this.runtimeError()
        );

        ipcRenderer.on('pyserver.stop__reply', (_, args) => (args==="success") ?
            this.notifyStop():this.runtimeError()
        );
    }

    createRuntime() {
        this.status = "creating";
        this.onCreate();

        ipcRenderer.send('pyenv.setup');
    }

    startRuntime() {
        this.status = "starting"
        ipcRenderer.send('pyserver.start')
    }

    stopRuntime() {
        this.status = "stopping"
        ipcRenderer.send('pyserver.stop')
    }

    runtimeError() {
        this.status = "error"
        this.onError();
    }

    notifySuccss() {
        this.status = "success"
        this.onSuccess();
    }

    notifyStop() {
        this.status = "stopped"
        this.onStop();
    }
}

