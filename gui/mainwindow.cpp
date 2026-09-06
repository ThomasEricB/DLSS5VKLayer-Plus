#include "mainwindow.h"
#include "passdialog.h"
#include "../common/runner_discovery.h"

#include <QAction>
#include <QActionGroup>
#include <QCheckBox>
#include <QCloseEvent>
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
#include <QIcon>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QPainter>
#include <QPixmap>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTextStream>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QVariantMap>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static QIcon gearIcon(const QWidget* w) {
    QIcon icon = QIcon::fromTheme("preferences-system-symbolic", QIcon::fromTheme("preferences-system"));
    if (!icon.isNull()) return icon;

    QPixmap pm(32, 32);
    pm.fill(Qt::transparent);
    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing);
    p.translate(16, 16);

    QColor c = w ? w->palette().color(QPalette::WindowText) : QColor(40, 40, 40);
    p.setPen(Qt::NoPen);
    p.setBrush(c);
    for (int i = 0; i < 8; ++i) {
        p.save();
        p.rotate(i * 45);
        p.drawRect(QRectF(-2.2, -15, 4.4, 7));
        p.restore();
    }
    p.drawEllipse(QPointF(0, 0), 10, 10);
    p.setCompositionMode(QPainter::CompositionMode_DestinationOut);
    p.drawEllipse(QPointF(0, 0), 4.5, 4.5);
    return QIcon(pm);
}

MainWindow::MainWindow(QWidget* parent) : QWidget(parent) {
    setWindowTitle("DLSS5VKLayer Helper");
    resize(420, 340);

    projectDir = findProjectDir();
    helperCliPath = findHelperCli();
    configFilePath = configPath();
    loadConfig();
    if (shmPath.isEmpty()) shmPath = qEnvironmentVariable("DLSSNR_SHM", defaultShmPath());
    if (logPath.isEmpty()) logPath = qEnvironmentVariable("DLSSNR_LOG", defaultLogPath());
    ensureShm();

    auto* root = new QVBoxLayout(this);

    settingsMenu = new QMenu(this);
    mvecAction = settingsMenu->addAction("Enable synthetic motion vectors");
    mvecAction->setCheckable(true);
    scaleMenu = settingsMenu->addMenu("Motion vector scale");
    scaleGroup = new QActionGroup(this);
    scaleGroup->setExclusive(true);
    scaleNormalizedAction = scaleMenu->addAction("Normalized [-1, 1]");
    scalePixelsAction = scaleMenu->addAction("Pixels");
    scaleUv01Action = scaleMenu->addAction("UV [0, 1]");
    for (QAction* a : { scaleNormalizedAction, scalePixelsAction, scaleUv01Action }) {
        a->setCheckable(true);
        scaleGroup->addAction(a);
    }
    qualityMenu = settingsMenu->addMenu("Motion vector quality");
    qualityGroup = new QActionGroup(this);
    qualityGroup->setExclusive(true);
    qualityFastAction = qualityMenu->addAction("Fast");
    qualityBalancedAction = qualityMenu->addAction("Balanced");
    qualityQualityAction = qualityMenu->addAction("Quality");
    for (QAction* a : { qualityFastAction, qualityBalancedAction, qualityQualityAction }) {
        a->setCheckable(true);
        qualityGroup->addAction(a);
    }
    settingsBtn = new QToolButton(this);
    settingsBtn->setAutoRaise(true);
    settingsBtn->setIcon(gearIcon(this));
    settingsBtn->setToolTip("Settings");
    settingsBtn->setPopupMode(QToolButton::InstantPopup);
    settingsBtn->setMenu(settingsMenu);

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
    presetCombo = new QComboBox(this);
    presetCombo->addItem("DLSS5 Native", uint32_t(DLSS5_PRESET_NATIVE));
    presetCombo->addItem("DLSS5 Natural", uint32_t(DLSS5_PRESET_NATURAL));
    presetCombo->addItem("DLSS5 Cinematic", uint32_t(DLSS5_PRESET_CINEMATIC));
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
        presetCombo->setCurrentIndex(int(ShmPreset(hdr)));
        intensitySpin->setValue(BitsToFloat(hdr->intensityBits.load()));
        localToneSpin->setValue(BitsToFloat(hdr->localToneBits.load()));
        localStructureSpin->setValue(BitsToFloat(hdr->localStructureBits.load()));
        skinStructureSpin->setValue(BitsToFloat(hdr->skinStructureBits.load()));
        sharpnessSpin->setValue(BitsToFloat(hdr->sharpnessBits.load()));
        mvecAction->setChecked(ShmMVecEnabled(hdr));
        setScaleAction(ShmMVecScaleMode(hdr));
        setQualityAction(ShmMVecQuality(hdr));
    } else {
        enabledBox->setChecked(true);
        passesSpin->setValue(1);
        presetCombo->setCurrentIndex(int(DLSS5_PRESET_NATIVE));
        intensitySpin->setValue(1.0);
        localToneSpin->setValue(1.0);
        localStructureSpin->setValue(1.0);
        skinStructureSpin->setValue(-1.0);
        sharpnessSpin->setValue(0.0);
        mvecAction->setChecked(true);
        setScaleAction(MVEC_SCALE_PIXELS);
        setQualityAction(MVEC_QUALITY_BALANCED);
    }

    form->addRow(enabledBox);
    form->addRow("DLSS5 preset", presetCombo);
    form->addRow("Passes", passesSpin);
    form->addRow("Intensity", intensitySpin);
    form->addRow("Local tone", localToneSpin);
    form->addRow("Local structure", localStructureSpin);
    form->addRow("Skin structure", skinStructureSpin);
    form->addRow("Sharpness", sharpnessSpin);
    root->addLayout(form);
    root->addStretch();

    auto* bottomRow = new QHBoxLayout;
    bottomRow->addStretch();
    bottomRow->addWidget(settingsBtn);
    root->addLayout(bottomRow);

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
    connect(presetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::writeControls);
    connect(mvecAction, &QAction::toggled, this, &MainWindow::writeControls);
    connect(scaleGroup, &QActionGroup::triggered, this, &MainWindow::writeControls);
    connect(qualityGroup, &QActionGroup::triggered, this, &MainWindow::writeControls);
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
    if (statusTimer) statusTimer->stop();

    if (statusProcess) {
        statusProcess->disconnect();
        statusProcess->setParent(nullptr);
        if (statusProcess->state() != QProcess::NotRunning) {
            statusProcess->kill();
            statusProcess->waitForFinished(1000);
        }
        delete statusProcess;
        statusProcess = nullptr;
    }

    if (helper) {
        helper->disconnect();
        helper->setParent(nullptr);
        if (helper->state() != QProcess::NotRunning) {
            helper->kill();
            helper->waitForFinished(1000);
        }
        delete helper;
        helper = nullptr;
    }

    if (shmBase) munmap(shmBase, 4096 + kMaxFrame * 2);
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (statusTimer) statusTimer->stop();
    if (hdr) hdr->quit.store(1);
    if (helperCliPath.isEmpty()) helperCliPath = findHelperCli();
    if (!helperCliPath.isEmpty()) QProcess::startDetached(helperCliPath, {"stop"});
    QWidget::closeEvent(event);
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
    const size_t total = 4096 + kMaxFrame * 2;
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
    if (hdr->magic.load() != kShmMagic || hdr->passes.load() == 0) ShmInitDefaults(hdr);
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

void MainWindow::setScaleAction(uint32_t mode) {
    if (mode == MVEC_SCALE_PIXELS) scalePixelsAction->setChecked(true);
    else if (mode == MVEC_SCALE_UV01) scaleUv01Action->setChecked(true);
    else scaleNormalizedAction->setChecked(true);
}

void MainWindow::setQualityAction(uint32_t quality) {
    if (quality == MVEC_QUALITY_FAST) qualityFastAction->setChecked(true);
    else if (quality == MVEC_QUALITY_QUALITY) qualityQualityAction->setChecked(true);
    else qualityBalancedAction->setChecked(true);
}

void MainWindow::writeControls() {
    if (!hdr) return;
    hdr->enabled.store(enabledBox->isChecked() ? 1u : 0u);
    hdr->passes.store(uint32_t(passesSpin->value()));
    hdr->preset.store(presetCombo->currentData().toUInt());
    hdr->intensityBits.store(FloatToBits(float(intensitySpin->value())));
    hdr->localToneBits.store(FloatToBits(float(localToneSpin->value())));
    hdr->localStructureBits.store(FloatToBits(float(localStructureSpin->value())));
    hdr->skinStructureBits.store(FloatToBits(float(skinStructureSpin->value())));
    hdr->sharpnessBits.store(FloatToBits(float(sharpnessSpin->value())));
    hdr->mvecEnabled.store(mvecAction->isChecked() ? 1u : 0u);
    hdr->mvecScaleMode.store(scalePixelsAction->isChecked() ? uint32_t(MVEC_SCALE_PIXELS)
        : scaleUv01Action->isChecked() ? uint32_t(MVEC_SCALE_UV01)
                                       : uint32_t(MVEC_SCALE_NORMALIZED));
    hdr->mvecQuality.store(qualityFastAction->isChecked() ? uint32_t(MVEC_QUALITY_FAST)
        : qualityQualityAction->isChecked() ? uint32_t(MVEC_QUALITY_QUALITY)
                                            : uint32_t(MVEC_QUALITY_BALANCED));
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