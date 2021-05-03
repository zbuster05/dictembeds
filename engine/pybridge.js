const { app, ipcMain } = require('electron')
const { execSync, exec } = require("child_process");

const fs = require('fs')
const fse = require('fs-extra')
var glob = require("glob")
const path = require('path')

const setupPyEnv = (event) => {
    let envPath = path.join(app.getPath('appData'), 'inscriptio', 'runtime');

    if (!fs.existsSync(envPath)) {
        fs.mkdirSync(envPath)
    }

    glob.sync(path.join(path.dirname(__filename), "inference", "!(env)")).map((file) => fse.copySync(file, path.join(envPath, path.basename(file))));

    exec(path.join(path.dirname(__filename), "inference", "setup.sh '")+envPath+"'", () => event.reply('pyenv.setup__reply', 'success'));
}

const checkForPyEnv = () => {
    let envPath = path.join(app.getPath('appData'), 'inscriptio', 'runtime');

    return (fs.existsSync(path.join(envPath, 'env')) && fs.existsSync(path.join(envPath, 'runtime.py')))
}

const startServer = () => {
    let envPath = path.join(app.getPath('appData'), 'inscriptio', 'runtime');

    return exec("source '"+path.join(envPath, "env", "bin", "activate")+"'; python '"+path.join(envPath, "runtime.py' server"));
}

ipcMain.on('pyenv.setup', (event, _) => {
    setupPyEnv(event);
});

ipcMain.on('pyenv.check', (event, _) => {
    let checkRes = checkForPyEnv();
    checkRes ? event.reply('pyenv.check__reply', 'success') : event.reply('pyenv.check__reply', 'failure');
});

ipcMain.on('pyserver.start', (event, _) => {
    let checkRes = checkForPyEnv();
    if (!checkRes) {
        event.reply('pyserver.start__reply', 'failure');
    }

    let process = startServer();
    event.reply('pyserver.start__reply', 'success');
})

module.exports = { setupPyEnv };
