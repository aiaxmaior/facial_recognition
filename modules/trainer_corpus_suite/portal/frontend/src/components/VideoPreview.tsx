import { useState, useRef, useEffect } from 'react'
import {
  Modal,
  Box,
  Group,
  Text,
  ActionIcon,
  Slider,
  Badge,
  Stack,
  Loader,
  Center,
  Tooltip
} from '@mantine/core'
import {
  IconPlayerPlay,
  IconPlayerPause,
  IconVolume,
  IconVolumeOff,
  IconMaximize,
  IconX,
  IconReload
} from '@tabler/icons-react'

interface VideoPreviewProps {
  opened: boolean
  onClose: () => void
  videoUrl: string
  title: string
  subtitle?: string
}

export function VideoPreview({ opened, onClose, videoUrl, title, subtitle }: VideoPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (opened && videoRef.current) {
      setIsLoading(true)
      setError(null)
      setCurrentTime(0)
      setIsPlaying(false)
    }
  }, [opened, videoUrl])

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleMuteToggle = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted
      setIsMuted(!isMuted)
    }
  }

  const handleSeek = (value: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = value
      setCurrentTime(value)
    }
  }

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
    }
  }

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration)
      setIsLoading(false)
    }
  }

  const handleError = () => {
    setError('Failed to load video')
    setIsLoading(false)
  }

  const handleFullscreen = () => {
    if (videoRef.current) {
      if (videoRef.current.requestFullscreen) {
        videoRef.current.requestFullscreen()
      }
    }
  }

  const handleReload = () => {
    if (videoRef.current) {
      setIsLoading(true)
      setError(null)
      videoRef.current.load()
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={
        <Stack gap={2}>
          <Text fw={600} size="lg">{title}</Text>
          {subtitle && <Text size="sm" c="dimmed">{subtitle}</Text>}
        </Stack>
      }
      size="xl"
      centered
      padding="md"
      styles={{
        body: { padding: 0 },
        header: { 
          background: 'linear-gradient(180deg, #1a1b1e 0%, #141517 100%)',
          borderBottom: '1px solid #2c2e33',
          padding: '12px 16px'
        },
        content: {
          background: '#0d1117',
          border: '1px solid #2c2e33'
        }
      }}
    >
      <Box>
        {/* Video Container */}
        <Box
          pos="relative"
          bg="#000"
          style={{
            aspectRatio: '16/9',
            borderRadius: 0
          }}
        >
          {isLoading && !error && (
            <Center pos="absolute" top={0} left={0} right={0} bottom={0} style={{ zIndex: 10 }}>
              <Stack align="center" gap="sm">
                <Loader color="cerulean" size="lg" type="bars" />
                <Text size="sm" c="dimmed">Loading video...</Text>
              </Stack>
            </Center>
          )}
          
          {error && (
            <Center pos="absolute" top={0} left={0} right={0} bottom={0} style={{ zIndex: 10 }}>
              <Stack align="center" gap="sm">
                <IconX size={48} color="#ff6b6b" />
                <Text c="red">{error}</Text>
                <ActionIcon variant="light" color="cerulean" onClick={handleReload}>
                  <IconReload size={18} />
                </ActionIcon>
              </Stack>
            </Center>
          )}

          <video
            ref={videoRef}
            src={videoUrl}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              display: error ? 'none' : 'block'
            }}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            onError={handleError}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onEnded={() => setIsPlaying(false)}
            onClick={handlePlayPause}
            playsInline
          />
        </Box>

        {/* Controls */}
        <Box
          p="sm"
          style={{
            background: 'linear-gradient(180deg, #141517 0%, #0d1117 100%)',
            borderTop: '1px solid #2c2e33'
          }}
        >
          {/* Progress Bar */}
          <Slider
            value={currentTime}
            min={0}
            max={duration || 100}
            step={0.1}
            onChange={handleSeek}
            size="xs"
            color="cerulean"
            mb="sm"
            styles={{
              track: { backgroundColor: '#2c2e33' },
              bar: { 
                background: 'linear-gradient(90deg, #0057b3, #3395f3)' 
              },
              thumb: {
                borderColor: '#3395f3',
                backgroundColor: '#3395f3'
              }
            }}
          />

          {/* Control Buttons */}
          <Group justify="space-between">
            <Group gap="xs">
              {/* Play/Pause */}
              <Tooltip label={isPlaying ? 'Pause' : 'Play'}>
                <ActionIcon
                  variant="filled"
                  color="cerulean"
                  size="lg"
                  onClick={handlePlayPause}
                  disabled={isLoading || !!error}
                >
                  {isPlaying ? <IconPlayerPause size={20} /> : <IconPlayerPlay size={20} />}
                </ActionIcon>
              </Tooltip>

              {/* Mute */}
              <Tooltip label={isMuted ? 'Unmute' : 'Mute'}>
                <ActionIcon
                  variant="subtle"
                  color="gray"
                  size="lg"
                  onClick={handleMuteToggle}
                  disabled={isLoading || !!error}
                >
                  {isMuted ? <IconVolumeOff size={18} /> : <IconVolume size={18} />}
                </ActionIcon>
              </Tooltip>

              {/* Time Display */}
              <Badge variant="light" color="dark" size="lg" radius="sm">
                <Text size="xs" ff="monospace">
                  {formatTime(currentTime)} / {formatTime(duration)}
                </Text>
              </Badge>
            </Group>

            <Group gap="xs">
              {/* Fullscreen */}
              <Tooltip label="Fullscreen">
                <ActionIcon
                  variant="subtle"
                  color="gray"
                  size="lg"
                  onClick={handleFullscreen}
                  disabled={isLoading || !!error}
                >
                  <IconMaximize size={18} />
                </ActionIcon>
              </Tooltip>
            </Group>
          </Group>
        </Box>
      </Box>
    </Modal>
  )
}
