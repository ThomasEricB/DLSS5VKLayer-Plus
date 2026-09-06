#include "mainwindow.h"
#include "passdialog.h"
#include "../common/runner_discovery.h"
#include "shm_binder.h"

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
#include <QGroupBox>
#include <QScrollArea>
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

    // The environment wins over the stored config, which is the order the layer and the helper both
    // use -- they read DLSSNR_SHM first and fall back. Having the interface do the opposite meant
    // pointing everything at one mapping and watching the interface report on another, with the path
    // it was actually using printed on screen the whole time.
    const QString shmEnv = qEnvironmentVariable("DLSSNR_SHM");
    if (!shmEnv.isEmpty()) shmPath = shmEnv;
    if (shmPath.isEmpty()) shmPath = defaultShmPath();

    const QString logEnv = qEnvironmentVariable("DLSSNR_LOG");
    if (!logEnv.isEmpty()) logPath = logEnv;
    if (logPath.isEmpty()) logPath = defaultLogPath();
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

    costLabel = new QLabel(this);
    costLabel->setWordWrap(true);
    root->addWidget(costLabel);

    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setWidget(buildSettings());
    root->addWidget(scroll, 1);

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

void MainWindow::updateStatus() {
    if (binder) binder->Reload();

    QString state = "shared memory not attached";
    if (hdr) {
        static const char* kStates[] = { "starting", "no Vulkan device", "no NGX binaries",
                                         "the model would not start", "running", "stopped" };
        const uint32_t hs = hdr->helperState.load();
        state = hs < 6 ? kStates[hs] : "unknown";
        const QString why = QString::fromStdString(
            ShmLoadString(hdr->helperReasonSeq, hdr->helperReason, kReasonBytes));
        if (!why.isEmpty()) state += " -- " + why;
    }

    statusLabel->setText(QString("Helper: %1\nProject: %2\nSHM: %3")
                             .arg(state, projectDir.isEmpty() ? "(not found)" : projectDir, shmPath));

    if (!hdr) {
        costLabel->clear();
        return;
    }

    const quint64 layerFrames = ShmLoad64(hdr->layerFramesLo, hdr->layerFramesHi);
    const quint64 modelFrames = ShmLoad64(hdr->helperFramesLo, hdr->helperFramesHi);
    const float ms = BitsToFloat(hdr->layerMsBits.load());
    const float measured = BitsToFloat(hdr->layerMeasuredWhiteBits.load());

    QString cost;
    if (hdr->layerCompositionUp.load() && layerFrames > 0)
        cost = QString("Composed %1 frames at %2 ms each; the model answered %3 of them with %4 pass(es).")
                   .arg(layerFrames)
                   .arg(double(ms), 0, 'f', 2)
                   .arg(modelFrames)
                   .arg(hdr->helperFeatures.load());
    else
        cost = "Waiting for a game to present through the layer.";

    if (measured > 0.0f)
        cost += QString("\nMeasured white point: %1.").arg(double(measured), 0, 'f', 3);

    costLabel->setText(cost);
}

// The settings panel.
//
// Grouped the way upstream groups them, because the grouping carries meaning: what a pass costs, how
// much of its answer lands, what the model itself was told, how colour is interpreted, and the tools
// for looking at the result. Controls under Model are marked as rebuilding the feature, because they
// are the ones that take a moment to appear.
QWidget* MainWindow::buildSettings() {
    auto* page = new QWidget(this);
    auto* col = new QVBoxLayout(page);
    binder = new ShmBinder(hdr, page);

    const auto group = [&](const QString& title) {
        auto* box = new QGroupBox(title, page);
        auto* form = new QFormLayout(box);
        col->addWidget(box);
        return form;
    };

    {
        auto* f = group("Neural rendering");
        binder->AddBool(f, "Enabled", &ShmHeader::enabled,
                        "Run the model at all. Off leaves the game's own frame untouched.");
    }
    {
        auto* f = group("Cost");
        binder->AddInt(f, "Passes", &ShmHeader::passes, 1, int(kMaxPasses),
                       "How many times the model runs over one frame, each pass shown the last one's "
                       "answer. Every pass is another full run of the model and another feature "
                       "holding its own history, so the cost is close to linear.",
                       ShmBinder::AtCreate);
        binder->AddBool(f, "Lift the pass limit", &ShmHeader::unlockPasses,
                        QString("Raises the ceiling from %1 to %2. Past a few passes the model is "
                                "enhancing its own output, which is outside what it was trained for.")
                            .arg(kDefaultMaxPasses)
                            .arg(kMaxPasses));
        binder->AddPercent(f, "Model resolution", &ShmHeader::workingScaleBits, 25, 200,
                           "What fraction of the frame the model works at. The frame itself is never "
                           "reduced. Below 100% also cuts what crosses shared memory, quadratically. "
                           "Above 100% the model supersamples, which on this transport is expensive: "
                           "at 200% on a 4K frame it is 132 MB each way, every frame.");
        binder->AddChoice(f, "Down-leg filter", &ShmHeader::scalingDownscaler,
                          { "(fsr1, unsupported)", "Bicubic", "Catmull-Rom", "Lanczos2", "Lanczos3",
                            "Kaiser2", "Kaiser3", "Magic" },
                          "How a supersampled answer is averaged back to the frame's size. Only used "
                          "above 100%.");
        passBtn = new QPushButton("Per-pass settings...", page);
        f->addRow(passBtn);
    }
    {
        auto* f = group("How much of it lands");
        binder->AddFloat(f, "Detail strength", &ShmHeader::transferStrengthBits, 0.0, 4.0, 0.05,
                         "How much of the model's edit reaches the frame. At zero the frame is "
                         "bit-identical to the game's own.");
        binder->AddFloat(f, "Colour strength", &ShmHeader::colourStrengthBits, 0.0, 4.0, 0.05,
                         "How much of the model's colour comes with its light. At zero the frame "
                         "keeps the game's hue exactly.");
        binder->AddFloat(f, "Highlight guard", &ShmHeader::maxRatioBits, 1.0, 30.0, 0.5,
                         "The most the pass may brighten or darken a pixel. A detail pass has no "
                         "business restyling a light source, whatever the model returns.");
        binder->AddChoice(f, "Enlargement", &ShmHeader::transfer, { "Classic", "Matched residual" },
                          "How a model that worked below the frame's size is brought back. Matched "
                          "residual carries only the model's difference up, so the two pictures being "
                          "composed are at the same scale.");
    }
    {
        auto* f = group("Model  (read when the model is built, so a change takes a moment)");
        binder->AddInt(f, "Preset", &ShmHeader::preset, 0, 15, "The model's own render preset.",
                       ShmBinder::AtCreate);
        binder->AddChoice(f, "Style", &ShmHeader::style, { "Default", "Natural", "Cinematic" },
                          "The model's own processing profiles.", ShmBinder::AtCreate);
        binder->AddFloat(f, "Intensity", &ShmHeader::intensityBits, 0.0, 4.0, 0.05,
                         "How hard the model works.", ShmBinder::AtCreate);
        binder->AddFloat(f, "Local structure", &ShmHeader::localStructureBits, 0.0, 4.0, 0.05, "",
                         ShmBinder::AtCreate);
        binder->AddFloat(f, "Local tone", &ShmHeader::localToneBits, 0.0, 4.0, 0.05, "",
                         ShmBinder::AtCreate);
        binder->AddFloat(f, "Skin structure", &ShmHeader::skinStructureBits, -1.0, 4.0, 0.05,
                         "-1 follows local structure, which is the model's own default. It is not a "
                         "strength of zero.",
                         ShmBinder::AtCreate);
        binder->AddBool(f, "Auto skin mask", &ShmHeader::autoMask, "The model's automatic skin mask.",
                        ShmBinder::AtCreate);
        binder->AddFloat(f, "Sharpness", &ShmHeader::sharpnessBits, 0.0, 1.0, 0.05,
                         "The one strength the model reads every frame, so it takes effect at once.");
    }
    {
        auto* f = group("Colour");
        binder->AddChoice(f, "Frame holds", &ShmHeader::colourMode,
                          { "Auto", "A finished picture", "Linear light" },
                          "Whether the swapchain carries a frame the game already tone mapped or "
                          "open-ended light. Auto decides from the format and is right for almost "
                          "every game.");
        binder->AddChoice(f, "White point from", &ShmHeader::whitePointSource,
                          { "The slider below", "Measured off the frame" },
                          "Only meaningful on a linear frame; a finished picture has no white point "
                          "to find.");
        binder->AddFloat(f, "Paper white", &ShmHeader::whitePointBits, 0.01, 2000.0, 0.1,
                         "What the model should treat as white, when it is not being measured.");
        binder->AddFloat(f, "White point scale", &ShmHeader::whitePointScaleBits, 0.01, 100.0, 0.05,
                         "Multiplies whichever white point is in use. Higher means highlights sit "
                         "lower on the curve.");
        binder->AddFloat(f, "Trim (measured)", &ShmHeader::whitePointTrimBits, 0.01, 100.0, 0.05,
                         "Multiplies a measured white point only. Kept apart from the slider because "
                         "a value found against one is meaningless against the other.");
    }
    {
        auto* f = group("Inspect");
        binder->AddBool(f, "Apply the model's edit", &ShmHeader::applyModel,
                        "Off keeps the whole pass running and shows the clean frame, so the cost is "
                        "unchanged and only the picture differs.");
        binder->AddBool(f, "Hold frame", &ShmHeader::holdFrame,
                        "Freeze the frame the pass works on, so changing a setting re-runs the model "
                        "and the composition on the same picture. The only clean way to compare two "
                        "settings.");
        binder->AddChoice(f, "Proxy", &ShmHeader::reversibleMode,
                          { "Soft knee", "Neutwo", "Neutwo, replace", "Hybrid", "Hybrid, replace" },
                          "Which picture the model is shown, and whether its answer is composed onto "
                          "the frame or substituted for it. Soft knee is the default and the two "
                          "replace modes are known to flash on bright lights.");
        binder->AddChoice(f, "Debug view", &ShmHeader::debugView,
                          { "Off", "The picture the model saw", "Its raw answer", "What it changed" },
                          "The last one is amplified and centred on grey, so both directions of the "
                          "edit are visible at once.");
        binder->AddFloat(f, "Debug scale", &ShmHeader::debugScaleBits, 0.01, 100.0, 0.1,
                         "What the debug views are multiplied by on their way out.");
        binder->AddChoice(f, "Compare", &ShmHeader::compareMode, { "Off", "Side by side", "Wipe" },
                          "Shows the pass against itself. The wipe cuts one frame and resamples "
                          "nothing, so it is the one to play with.");
        binder->AddFloat(f, "Split", &ShmHeader::compareSplitBits, 0.0, 1.0, 0.01, "");
        binder->AddFloat(f, "Zoom", &ShmHeader::compareZoomBits, 1.0, 2.0, 0.05,
                         "Side by side only. 1 fits the whole frame and accepts the bars; 2 fills the "
                         "half and crops.");
        binder->AddBool(f, "Swap sides", &ShmHeader::compareSwap,
                        "Which side the edited frame sits on. Worth having because the eye is not "
                        "even-handed about left and right.");

        auto* capRow = new QHBoxLayout;
        captureFrames = new QSpinBox(page);
        captureFrames->setRange(1, 64);
        captureFrames->setValue(8);
        captureBtn = new QPushButton("Capture frames", page);
        captureBtn->setToolTip("Writes that many matched before/after pairs to the state directory. "
                               "Same frames, same run, one variable.");
        capRow->addWidget(captureFrames);
        capRow->addWidget(captureBtn);
        f->addRow(capRow);

        connect(captureBtn, &QPushButton::clicked, this, [this] {
            if (!hdr) return;
            hdr->captureRequest.store(uint32_t(captureFrames->value()));
            hdr->controlSeq.fetch_add(1);
        });
    }

    connect(passBtn, &QPushButton::clicked, this, [this] {
        if (!hdr) return;
        PassDialog dlg(hdr, this);
        dlg.exec();
    });

    col->addStretch(1);
    binder->Reload();
    return page;
}
