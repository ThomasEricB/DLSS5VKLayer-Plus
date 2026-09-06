#include "mainwindow.h"
#include "passdialog.h"
#include "../common/runner_discovery.h"

#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTextStream>
#include <QTimer>
#include <QVBoxLayout>
#include <QVariantMap>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

MainWindow::MainWindow(QWidget* parent) : QWidget(parent) {
    setWindowTitle("DLSS5VKLayer Helper");
    resize(420, 300);

    projectDir = findProjectDir();
    helperCliPath = findHelperCli();
    configFilePath = configPath();
    loadConfig();
    if (shmPath.isEmpty()) shmPath = qEnvironmentVariable("DLSSNR_SHM", defaultShmPath());
    if (logPath.isEmpty()) logPath = qEnvironmentVariable("DLSSNR_LOG", defaultLogPath());
    ensureShm();

    auto* root = new QVBoxLayout(this);

    statusLabel = new QLabel(this);
    root->addWidget(statusLabel);

    auto* runnerForm = new QFormLayout;
    runnerCombo = new QComboBox(this);
    runnerPathEdit = new QLineEdit(this);
    browseRunnerBtn = new QPushButton("Browse...", this);
    auto* runnerPathRow = new QHBoxLayout;
    runnerPathRow->addWidget(runnerPathEdit);
    runnerPathRow->addWidget(browseRunnerBtn);
    runnerForm->addRow("Runner", runnerCombo);
    runnerForm->addRow("Path", runnerPathRow);
    root->addLayout(runnerForm);

    auto* buttons = new QHBoxLayout;
    startBtn = new QPushButton("Start helper", this);
    stopBtn = new QPushButton("Stop helper", this);
    buttons->addWidget(startBtn);
    buttons->addWidget(stopBtn);
    root->addLayout(buttons);

    passBtn = new QPushButton("Per-pass settings...", this);
    root->addWidget(passBtn);

    auto* form = new QFormLayout;
    enabledBox = new QCheckBox("Neural enabled", this);
    passesSpin = new QSpinBox(this);
    passesSpin->setRange(1, int(kMaxPasses));
    intensitySpin = new QDoubleSpinBox(this);
    intensitySpin->setRange(0.0, 4.0);
    intensitySpin->setSingleStep(0.05);
    localToneSpin = new QDoubleSpinBox(this);
    localToneSpin->setRange(0.0, 4.0);
    localToneSpin->setSingleStep(0.05);
    localStructureSpin = new QDoubleSpinBox(this);
    localStructureSpin->setRange(0.0, 4.0);
    localStructureSpin->setSingleStep(0.05);
    skinStructureSpin = new QDoubleSpinBox(this);
    skinStructureSpin->setRange(-1.0, 4.0);
    skinStructureSpin->setSingleStep(0.05);
    sharpnessSpin = new QDoubleSpinBox(this);
    sharpnessSpin->setRange(0.0, 1.0);
    sharpnessSpin->setSingleStep(0.05);

    if (hdr) {
        enabledBox->setChecked(ShmNeuralEnabled(hdr));
        passesSpin->setValue(int(ShmPasses(hdr)));
        intensitySpin->setValue(BitsToFloat(hdr->intensityBits.load()));
        localToneSpin->setValue(BitsToFloat(hdr->localToneBits.load()));
        localStructureSpin->setValue(BitsToFloat(hdr->localStructureBits.load()));
        skinStructureSpin->setValue(BitsToFloat(hdr->skinStructureBits.load()));
        sharpnessSpin->setValue(BitsToFloat(hdr->sharpnessBits.load()));
    } else {
        enabledBox->setChecked(true);
        passesSpin->setValue(1);
        intensitySpin->setValue(1.0);
        localToneSpin->setValue(1.0);
        localStructureSpin->setValue(1.0);
        skinStructureSpin->setValue(-1.0);
        sharpnessSpin->setValue(0.0);
    }

    form->addRow(enabledBox);
    form->addRow("Passes", passesSpin);
    form->addRow("Intensity", intensitySpin);
    form->addRow("Local tone", localToneSpin);
    form->addRow("Local structure", localStructureSpin);
    form->addRow("Skin structure", skinStructureSpin);
    form->addRow("Sharpness", sharpnessSpin);
    root->addLayout(form);

    connect(startBtn, &QPushButton::clicked, this, &MainWindow::startHelper);
    connect(stopBtn, &QPushButton::clicked, this, &MainWindow::stopHelper);
    connect(passBtn, &QPushButton::clicked, this, [this] {
        if (!hdr) return;
        PassDialog dlg(hdr, this);
        dlg.exec();
    });
    connect(runnerCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::applyRunnerSelection);
    connect(runnerPathEdit, &QLineEdit::editingFinished, this, [this] {
        runnerPath = runnerPathEdit->text().trimmed();
        if (!runnerPath.isEmpty()) {
            runnerType = runnerPath.contains("proton", Qt::CaseInsensitive) ? "proton" : "wine";
            saveConfig();
        }
    });
    connect(browseRunnerBtn, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(this, "Select runner", QDir::homePath());
        if (path.isEmpty()) return;
        runnerPath = path;
        runnerType = runnerPath.contains("proton", Qt::CaseInsensitive) ? "proton" : "wine";
        runnerPathEdit->setText(runnerPath);
        saveConfig();
    });
    connect(enabledBox, &QCheckBox::toggled, this, &MainWindow::writeControls);
    connect(passesSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, &MainWindow::writeControls);
    connect(intensitySpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::writeControls);
    connect(localToneSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::writeControls);
    connect(localStructureSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::writeControls);
    connect(skinStructureSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::writeControls);
    connect(sharpnessSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &MainWindow::writeControls);

    populateRunners();

    statusTimer = new QTimer(this);
    connect(statusTimer, &QTimer::timeout, this, &MainWindow::updateStatus);
    statusTimer->start(1000);
    updateStatus();
}

MainWindow::~MainWindow() {
    if (helper) {
        // Let the helper keep running if the GUI is closed.
        helper->setParent(nullptr);
    }
    if (shmBase) munmap(shmBase, ShmTotalBytes());
}

QString MainWindow::findProjectDir() const {
    QStringList candidates;
    candidates << QDir::currentPath() << QCoreApplication::applicationDirPath();
    QDir d(QCoreApplication::applicationDirPath());
    for (int i = 0; i < 6; ++i) {
        candidates << d.absolutePath();
        if (!d.cdUp()) break;
    }
    for (const QString& c : candidates) {
        if (QFile::exists(c + "/build/dlssnr_helper.exe")) return c;
    }
    return QDir::currentPath();
}

QString MainWindow::findHelperCli() const {
    const QString env = qEnvironmentVariable("DLSSNR_HELPER_CLI");
    if (!env.isEmpty() && QFile::exists(env)) return env;

    const QString found = QStandardPaths::findExecutable("dlssnr-helper");
    if (!found.isEmpty()) return found;

    const QStringList candidates = {
        projectDir + "/dlssnr-helper",
        QDir::homePath() + "/.local/bin/dlssnr-helper",
        "/usr/bin/dlssnr-helper",
        "/usr/local/bin/dlssnr-helper"
    };
    for (const QString& c : candidates) {
        if (QFile::exists(c)) return c;
    }
    return QString();
}

QString MainWindow::configPath() const {
    const QString base = qEnvironmentVariable("XDG_CONFIG_HOME", QDir::homePath() + "/.config");
    return base + "/dlssnr/config.ini";
}

QString MainWindow::defaultShmPath() const {
    QString rt = qEnvironmentVariable("XDG_RUNTIME_DIR");
    if (rt.isEmpty()) rt = QString("/tmp/dlssnr-%1").arg(::getuid());
    return rt + "/dlssnr/shm.bin";
}

QString MainWindow::defaultLogPath() const {
    const QString base = qEnvironmentVariable("XDG_STATE_HOME", QDir::homePath() + "/.local/state");
    return base + "/dlssnr/helper.log";
}

void MainWindow::loadConfig() {
    QFile f(configFilePath);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return;

    QTextStream in(&f);
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty() || line.startsWith('#') || !line.contains('=')) continue;
        const QString key = line.section('=', 0, 0).trimmed();
        QString value = line.section('=', 1).trimmed();
        if (value.size() >= 2 && value.startsWith('"') && value.endsWith('"')) {
            value = value.mid(1, value.size() - 2);
        }
        if (key == "runner_type") runnerType = value;
        else if (key == "runner_path") runnerPath = value;
        else if (key == "binaries") binariesPath = value;
        else if (key == "shm") shmPath = value;
        else if (key == "log") logPath = value;
        else if (key == "dxvk_vendor") dxvkVendor = value;
        else if (key == "dxvk_device") dxvkDevice = value;
    }
}

void MainWindow::saveConfig() {
    const QFileInfo info(configFilePath);
    QDir().mkpath(info.absolutePath());

    QFile f(configFilePath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) return;

    QTextStream out(&f);
    out << "runner_type=" << runnerType << "\n";
    out << "runner_path=" << runnerPath << "\n";
    out << "binaries=" << binariesPath << "\n";
    out << "shm=" << shmPath << "\n";
    out << "log=" << logPath << "\n";
    out << "dxvk_vendor=" << dxvkVendor << "\n";
    out << "dxvk_device=" << dxvkDevice << "\n";
}

void MainWindow::populateRunners() {
    const QSignalBlocker blocker(runnerCombo);
    runnerCombo->clear();

    const auto runners = dlssnr::discoverCustomRunners();
    for (const auto& r : runners) {
        QVariantMap data;
        data["type"] = "proton";
        data["path"] = QString::fromStdString(r.path);
        runnerCombo->addItem(QString::fromStdString(dlssnr::runnerDisplayName(r)), data);
    }

    const QString wine = QStandardPaths::findExecutable("wine");
    if (!wine.isEmpty()) {
        QVariantMap data;
        data["type"] = "wine";
        data["path"] = wine;
        runnerCombo->addItem("System Wine", data);
    }

    bool selected = false;
    for (int i = 0; i < runnerCombo->count(); ++i) {
        if (runnerCombo->itemData(i).toMap().value("path").toString() == runnerPath) {
            runnerCombo->setCurrentIndex(i);
            selected = true;
            break;
        }
    }
    if (!selected && !runnerPath.isEmpty()) {
        QVariantMap data;
        data["type"] = runnerPath.contains("proton", Qt::CaseInsensitive) ? "proton" : "wine";
        data["path"] = runnerPath;
        runnerCombo->addItem("Custom: " + runnerPath, data);
        runnerCombo->setCurrentIndex(runnerCombo->count() - 1);
        selected = true;
    }
    if (!selected && runnerCombo->count() > 0) {
        runnerCombo->setCurrentIndex(0);
        applyRunnerSelection(0);
    }

    runnerPathEdit->setText(runnerPath);
}

void MainWindow::applyRunnerSelection(int index) {
    if (index < 0) return;
    const QVariantMap data = runnerCombo->itemData(index).toMap();
    runnerType = data.value("type").toString();
    runnerPath = data.value("path").toString();
    runnerPathEdit->setText(runnerPath);
    saveConfig();
}

bool MainWindow::ensureShm() {
    if (hdr) return true;
    const QByteArray p = shmPath.toUtf8();
    int fd = open(p.constData(), O_RDWR | O_CREAT, 0666);
    if (fd < 0) return false;
    const size_t total = ShmTotalBytes();
    struct stat st{};
    if (fstat(fd, &st) != 0 || size_t(st.st_size) < total) {
        if (ftruncate(fd, off_t(total)) != 0) {
            ::close(fd);
            return false;
        }
    }
    void* m = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (m == MAP_FAILED) return false;
    shmBase = m;
    hdr = (ShmHeader*)m;
    if (hdr->magic.load() != kShmMagic || hdr->version.load() != kShmVersion ||
        hdr->passes.load() == 0)
        ShmInitDefaults(hdr);
    return true;
}

void MainWindow::startHelper() {
    if (helperCliPath.isEmpty()) helperCliPath = findHelperCli();
    if (helperCliPath.isEmpty()) {
        QMessageBox::warning(this, "DLSS5VKLayer", "dlssnr-helper CLI not found.");
        return;
    }
    if (!ensureShm()) {
        QMessageBox::warning(this, "DLSS5VKLayer", "Could not open shared memory file.");
        return;
    }
    if (hdr) hdr->quit.store(0);
    saveConfig();
    QProcess::startDetached(helperCliPath, {"start"});
    updateStatus();
}

void MainWindow::stopHelper() {
    if (helperCliPath.isEmpty()) helperCliPath = findHelperCli();
    if (hdr) hdr->quit.store(1);
    if (!helperCliPath.isEmpty()) {
        QProcess::startDetached(helperCliPath, {"stop"});
    }
    updateStatus();
}

void MainWindow::writeControls() {
    if (!hdr) return;
    hdr->enabled.store(enabledBox->isChecked() ? 1u : 0u);
    hdr->passes.store(uint32_t(passesSpin->value()));
    hdr->intensityBits.store(FloatToBits(float(intensitySpin->value())));
    hdr->localToneBits.store(FloatToBits(float(localToneSpin->value())));
    hdr->localStructureBits.store(FloatToBits(float(localStructureSpin->value())));
    hdr->skinStructureBits.store(FloatToBits(float(skinStructureSpin->value())));
    hdr->sharpnessBits.store(FloatToBits(float(sharpnessSpin->value())));
    hdr->controlSeq.fetch_add(1);
}

void MainWindow::updateStatus() {
    if (helperCliPath.isEmpty()) helperCliPath = findHelperCli();
    if (helperCliPath.isEmpty()) {
        statusLabel->setText("Helper CLI not found");
        startBtn->setEnabled(false);
        stopBtn->setEnabled(false);
        return;
    }

    if (!statusProcess) {
        statusProcess = new QProcess(this);
        connect(statusProcess, &QProcess::finished, this, [this] {
            const QString out = QString::fromUtf8(statusProcess->readAllStandardOutput());
            const QStringList lines = out.split('\n', Qt::SkipEmptyParts);
            const QString first = lines.isEmpty() ? "unknown" : lines.first();
            helperRunning = first.startsWith("helper running");
            statusLabel->setText(first);
            startBtn->setEnabled(!helperRunning);
            stopBtn->setEnabled(helperRunning);
        });
    }

    if (statusProcess->state() != QProcess::NotRunning) return;
    statusProcess->setProgram(helperCliPath);
    statusProcess->setArguments({"status"});
    statusProcess->start();
}