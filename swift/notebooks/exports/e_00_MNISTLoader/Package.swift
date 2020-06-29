// swift-tools-version:4.2
import PackageDescription

let package = Package(
name: "e_00_MNISTLoader",
products: [
.library(name: "e_00_MNISTLoader", targets: ["e_00_MNISTLoader"]),

],
dependencies: [
.package(url: "https://github.com/mxcl/Path.swift", from: "0.16.1"),
    .package(url: "https://github.com/saeta/Just", from: "0.7.2"),
    .package(url: "https://github.com/latenitesoft/NotebookExport", from: "0.5.0")
],
targets: [
.target(name: "e_00_MNISTLoader", dependencies: ["Path", "Just", "NotebookExport"]),

]
)